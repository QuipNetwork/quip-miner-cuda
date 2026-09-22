//! Nonce-parallel probe screen (`quip-screen`).
//!
//! Ranks `PoW` nonces by a short anneal on the GPU so the coordinator can send
//! only the deepest ones to the full miner. See `src/screen.rs`.
//!
//! ```text
//! quip-screen --spec topology.json verify    # device draw and scoring == protocol
//! quip-screen --spec topology.json bench     # probe rate, optional per-nonce energies
//! quip-screen --spec topology.json serve     # coordinator sidecar on stdin/stdout
//! ```
//!
//! The spec is a topology JSON (`nodes`, `edges`, `allowed_h_milli`,
//! `allowed_j_milli`), the drive/seed-chain format. `edges` must be in the
//! chain's order, because the nonce draw assigns couplings in that order.

use clap::{Parser, Subcommand};
use quip_miner_cuda::screen::{
    reference_energy, reference_masks, Screen, ScreenParams, ScreenTopology,
};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::process::ExitCode;
use std::sync::mpsc;
use std::time::{Instant, SystemTime};

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Topology spec JSON in the chain's edge order.
    #[arg(long, global = true)]
    spec: Option<String>,
    /// CUDA device index.
    #[arg(long, global = true, default_value_t = 0)]
    device: usize,
    /// Reads per nonce = `2^reads_log2` (2..=5).
    #[arg(long, global = true, default_value_t = 2)]
    reads_log2: u32,
    /// Sweeps per probe.
    #[arg(long, global = true, default_value_t = 512)]
    sweeps: usize,
    /// Groups per launch = SMs x waves.
    #[arg(long, global = true, default_value_t = 16)]
    waves: usize,
    /// Threads per block.
    #[arg(long, global = true, default_value_t = 256)]
    threads: u32,
}

#[derive(Subcommand)]
enum Command {
    /// Check the device couplings and energies against the protocol draw and scoring.
    Verify,
    /// Measure the probe rate; optionally write `<nonce hex> <energy>` lines.
    Bench {
        /// One 64-hex nonce per line; random nonces if absent.
        #[arg(long)]
        nonces: Option<String>,
        /// Random nonces to screen when no file is given.
        #[arg(long, default_value_t = 50_000)]
        count: usize,
        /// Output file for per-nonce energies.
        #[arg(long)]
        out: Option<String>,
    },
    /// Coordinator sidecar. stdin: `R <generation> <prev hash hex> <identity hex>`
    /// starts a round, `Q` quits. stdout: `K <generation> <salt hex> <energy>` for
    /// kept nonces and `S <nonces/s> <cutoff> <kept/s> r<reads>x<sweeps>` every 10 s.
    Serve {
        /// Kept nonces per second to hand to the coordinator.
        #[arg(long, default_value_t = 40.0)]
        keep_per_s: f64,
        /// JSON file re-read every 5 s: `keep_per_s`, `sweeps`, `reads_log2`,
        /// `waves`, `threads` (any subset).
        #[arg(long)]
        tune: Option<String>,
    },
}

struct Spec {
    topo: ScreenTopology,
    couplings_milli: Vec<i32>,
}

fn load_spec(path: &str) -> Result<Spec, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let ints = |key: &str| -> Result<Vec<i64>, String> {
        v[key]
            .as_array()
            .ok_or_else(|| format!("{path}: missing {key}"))?
            .iter()
            .map(|x| {
                x.as_i64()
                    .ok_or_else(|| format!("{path}: non-integer in {key}"))
            })
            .collect()
    };
    let nodes = ints("nodes")?;
    let index: HashMap<i64, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let edges = v["edges"]
        .as_array()
        .ok_or_else(|| format!("{path}: missing edges"))?
        .iter()
        .map(|e| {
            let end = |k: usize| {
                e[k].as_i64()
                    .and_then(|id| index.get(&id).copied())
                    .ok_or_else(|| format!("{path}: edge {e} names an unknown node"))
            };
            Ok((end(0)?, end(1)?))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let narrow = |xs: Vec<i64>| -> Result<Vec<i32>, String> {
        xs.into_iter()
            .map(|x| i32::try_from(x).map_err(|_| format!("{path}: value {x} out of range")))
            .collect()
    };
    let fields_milli = narrow(ints("allowed_h_milli")?)?;
    let couplings_milli = narrow(ints("allowed_j_milli")?)?;
    let topo = ScreenTopology::new(nodes.len(), edges, &fields_milli, &couplings_milli)
        .map_err(|e| format!("{path}: {e}"))?;
    Ok(Spec {
        topo,
        couplings_milli,
    })
}

fn from_hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            write!(s, "{b:02x}").expect("writing to a String cannot fail");
            s
        })
}

/// xorshift64 for test nonces and the per-process salt prefix; not security relevant.
struct XorShift(u64);

impl XorShift {
    fn seeded() -> Self {
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "only the low bits seed the generator"
        )]
        let low = t as u64;
        Self(low ^ 0x9E37_79B9_7F4A_7C15 ^ u64::from(std::process::id()) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn bytes32(&mut self) -> [u8; 32] {
        let mut b = [0u8; 32];
        for chunk in b.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        b
    }
}

fn params(cli: &Cli) -> ScreenParams {
    ScreenParams {
        reads_log2: cli.reads_log2,
        sweeps: cli.sweeps,
        waves: cli.waves,
        threads: cli.threads,
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let Some(path) = cli.spec.as_deref() else {
        eprintln!("quip-screen: --spec <topology.json> is required");
        return ExitCode::from(64);
    };
    let spec = match load_spec(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("quip-screen: {e}");
            return ExitCode::from(64);
        }
    };
    let result = match &cli.command {
        Command::Verify => verify(&cli, &spec),
        Command::Bench { nonces, count, out } => {
            bench(&cli, &spec, nonces.as_deref(), *count, out.as_deref())
        }
        Command::Serve { keep_per_s, tune } => serve(&cli, &spec, *keep_per_s, tune.as_deref()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("quip-screen: {e}");
            ExitCode::FAILURE
        }
    }
}

fn verify(cli: &Cli, spec: &Spec) -> Result<(), String> {
    let mut ok = true;
    let mut rng = XorShift::seeded();
    for reads_log2 in [2u32, 3] {
        let mut p = params(cli);
        p.reads_log2 = reads_log2;
        p.waves = 1;
        let mut screen = Screen::open(cli.device, &spec.topo, p).map_err(|e| e.to_string())?;
        let nonces: Vec<[u8; 32]> = (0..screen.batch()).map(|_| rng.bytes32()).collect();
        let run = screen.run(&nonces, 7, true).map_err(|e| e.to_string())?;
        let (masks, state) = run.group0_debug.ok_or("no debug output")?;
        let npg = 32usize >> reads_log2;
        let want = reference_masks(
            &nonces[..npg],
            reads_log2,
            &spec.topo,
            &spec.couplings_milli,
        )
        .map_err(|e| e.to_string())?;
        let bad_masks = masks.iter().zip(&want).filter(|(a, b)| a != b).count();
        let mut bad_energy = 0;
        for (lane, &e) in run.group0_lanes.iter().enumerate() {
            let cpu = reference_energy(
                nonces[lane % npg],
                &state,
                lane,
                &spec.topo,
                &spec.couplings_milli,
            )
            .map_err(|e| e.to_string())?;
            if i64::from(e) != cpu {
                bad_energy += 1;
            }
        }
        let min_ok = (0..npg).all(|k| {
            Some(run.best[k]) == run.group0_lanes.iter().skip(k).step_by(npg).copied().min()
        });
        println!(
            "reads {}: couplings {bad_masks}/{} edges differ, lane energies {bad_energy}/32 differ, per-nonce min {}",
            1 << reads_log2,
            want.len(),
            if min_ok { "ok" } else { "WRONG" }
        );
        ok &= bad_masks == 0 && bad_energy == 0 && min_ok;
    }
    println!("{}", if ok { "VERIFY_OK" } else { "VERIFY_FAILED" });
    if ok {
        Ok(())
    } else {
        Err("device results differ from the protocol".into())
    }
}

#[expect(clippy::cast_precision_loss, reason = "rates are display values")]
fn bench(
    cli: &Cli,
    spec: &Spec,
    file: Option<&str>,
    count: usize,
    out: Option<&str>,
) -> Result<(), String> {
    let mut screen =
        Screen::open(cli.device, &spec.topo, params(cli)).map_err(|e| e.to_string())?;
    let nonces: Vec<[u8; 32]> = if let Some(p) = file {
        std::fs::read_to_string(p)
            .map_err(|e| format!("{p}: {e}"))?
            .lines()
            .filter_map(from_hex32)
            .collect()
    } else {
        let mut rng = XorShift::seeded();
        (0..count).map(|_| rng.bytes32()).collect()
    };
    if nonces.is_empty() {
        return Err("no nonces".into());
    }
    let batch = screen.batch();
    // Warm-up launch (JIT, clocks), excluded from the rate.
    screen
        .run(&nonces[..batch.min(nonces.len())], 1, false)
        .map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let mut energies = Vec::with_capacity(nonces.len());
    for (k, chunk) in nonces.chunks(batch).enumerate() {
        let seed = u32::try_from(k).unwrap_or(u32::MAX).wrapping_add(100);
        energies.extend(
            screen
                .run(chunk, seed, false)
                .map_err(|e| e.to_string())?
                .best,
        );
    }
    let secs = t0.elapsed().as_secs_f64();
    let mut sorted = energies.clone();
    sorted.sort_unstable();
    println!(
        "rate {:.0} nonces/s ({} nonces, {} reads x {} sweeps, batch {batch}) | median {} deepest {}",
        energies.len() as f64 / secs,
        energies.len(),
        1 << cli.reads_log2,
        cli.sweeps,
        sorted[sorted.len() / 2],
        sorted[0]
    );
    if let Some(p) = out {
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(p).map_err(|e| format!("{p}: {e}"))?);
        for (n, e) in nonces.iter().zip(&energies) {
            writeln!(f, "{} {e}", to_hex(n)).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

type Round = (u64, [u8; 32], [u8; 32]);

fn read_rounds(tx: &mpsc::Sender<Option<Round>>) {
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["R", generation, prev, identity] => {
                if let (Ok(g), Some(p), Some(i)) =
                    (generation.parse(), from_hex32(prev), from_hex32(identity))
                {
                    if tx.send(Some((g, p, i))).is_err() {
                        return;
                    }
                }
            }
            ["Q"] => break,
            _ => {}
        }
    }
    // The serve loop may already be gone; nothing is left to tell.
    tx.send(None).unwrap_or_default();
}

/// Running histogram of probe energies; the cutoff passes a target share.
struct Cutoff {
    counts: Vec<f64>,
    total: f64,
}

impl Cutoff {
    const OFFSET: i64 = 20_000;

    fn new() -> Self {
        Self {
            counts: vec![0.0; 40_000],
            total: 0.0,
        }
    }

    fn add(&mut self, energies: &[i32]) {
        // Halve old counts so the histogram tracks the last ~2M nonces.
        if self.total > 2e6 {
            self.counts.iter_mut().for_each(|c| *c *= 0.5);
            self.total *= 0.5;
        }
        let top = i64::try_from(self.counts.len()).unwrap_or(i64::MAX) - 1;
        for &e in energies {
            let i = usize::try_from((i64::from(e) + Self::OFFSET).clamp(0, top)).unwrap_or(0);
            self.counts[i] += 1.0;
            self.total += 1.0;
        }
    }

    /// Highest energy whose cumulative share stays within `share`.
    fn at_share(&self, share: f64) -> i64 {
        let mut acc = 0.0;
        for (i, &c) in self.counts.iter().enumerate() {
            if (acc + c) / self.total.max(1.0) > share {
                return i64::try_from(i).unwrap_or(0) - Self::OFFSET - 1;
            }
            acc += c;
        }
        i64::MAX
    }
}

#[expect(clippy::cast_precision_loss, reason = "rates are display values")]
fn serve(cli: &Cli, spec: &Spec, keep_per_s: f64, tune: Option<&str>) -> Result<(), String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || read_rounds(&tx));
    let mut keep_per_s = keep_per_s;
    let mut p = params(cli);
    let mut screen = Screen::open(cli.device, &spec.topo, p).map_err(|e| e.to_string())?;
    let mut rng = XorShift::seeded();
    // Salt: 0x5c + 7 random bytes fixed per process, then a u64 LE counter,
    // then zeros. The coordinator's own counter salts leave bytes 8.. zero, so
    // the two never collide.
    let mut prefix = rng.bytes32();
    prefix[0] = 0x5c;
    let mut counter: u64 = 0;
    let mut round: Option<Round> = None;
    let mut cutoff = Cutoff::new();
    let mut rate = 0.0f64;
    let mut tune_seen: Option<SystemTime> = None;
    let mut tune_checked = Instant::now();
    let (mut window_n, mut window_kept, mut window_t) = (0usize, 0usize, Instant::now());
    loop {
        if round.is_none() {
            match rx.recv() {
                Ok(Some(r)) => round = Some(r),
                _ => return Ok(()),
            }
        }
        while let Ok(msg) = rx.try_recv() {
            match msg {
                Some(r) => round = Some(r),
                None => return Ok(()),
            }
        }
        let Some((generation, prev, identity)) = round else {
            continue;
        };
        if let Some(path) = tune {
            if tune_checked.elapsed().as_secs() >= 5 {
                tune_checked = Instant::now();
                if let Some(next) = retune(path, &mut tune_seen, p, &mut keep_per_s) {
                    match Screen::open(cli.device, &spec.topo, next) {
                        Ok(s) => {
                            screen = s;
                            p = next;
                            cutoff = Cutoff::new();
                            rate = 0.0;
                        }
                        Err(e) => eprintln!("quip-screen: tune rejected: {e}"),
                    }
                }
            }
        }
        let batch = screen.batch();
        let mut salts = Vec::with_capacity(batch);
        let mut nonces = Vec::with_capacity(batch);
        for _ in 0..batch {
            counter += 1;
            let mut salt = [0u8; 32];
            salt[..8].copy_from_slice(&prefix[..8]);
            salt[8..16].copy_from_slice(&counter.to_le_bytes());
            nonces.push(quip_protocol::derive::derive_nonce(prev, identity, salt));
            salts.push(salt);
        }
        let t = Instant::now();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the launch seed only needs to vary"
        )]
        let seed = counter as u32;
        let best = screen
            .run(&nonces, seed, false)
            .map_err(|e| e.to_string())?
            .best;
        let batch_rate = batch as f64 / t.elapsed().as_secs_f64().max(1e-6);
        rate = if rate == 0.0 {
            batch_rate
        } else {
            0.9 * rate + 0.1 * batch_rate
        };
        cutoff.add(&best);
        let limit = cutoff.at_share((keep_per_s / rate.max(1.0)).min(1.0));
        let mut out = std::io::stdout().lock();
        for (salt, &e) in salts.iter().zip(&best) {
            if i64::from(e) <= limit {
                writeln!(out, "K {generation} {} {e}", to_hex(salt)).map_err(|e| e.to_string())?;
                window_kept += 1;
            }
        }
        window_n += batch;
        let elapsed = window_t.elapsed().as_secs_f64();
        if elapsed >= 10.0 {
            writeln!(
                out,
                "S {:.0} {limit} {:.1} r{}x{}",
                window_n as f64 / elapsed,
                window_kept as f64 / elapsed,
                1u32 << p.reads_log2,
                p.sweeps
            )
            .map_err(|e| e.to_string())?;
            (window_n, window_kept, window_t) = (0, 0, Instant::now());
        }
        out.flush().map_err(|e| e.to_string())?;
    }
}

/// Re-read the tune file when it changed. Returns new launch parameters when
/// they differ from `current`; updates `keep_per_s` in place.
fn retune(
    path: &str,
    seen: &mut Option<SystemTime>,
    current: ScreenParams,
    keep_per_s: &mut f64,
) -> Option<ScreenParams> {
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok()?;
    if *seen == Some(modified) {
        return None;
    }
    *seen = Some(modified);
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    if let Some(k) = v["keep_per_s"].as_f64() {
        *keep_per_s = k;
    }
    let pick = |key: &str| v[key].as_u64();
    let next = ScreenParams {
        reads_log2: pick("reads_log2")
            .and_then(|x| u32::try_from(x).ok())
            .unwrap_or(current.reads_log2),
        sweeps: pick("sweeps")
            .and_then(|x| usize::try_from(x).ok())
            .unwrap_or(current.sweeps),
        waves: pick("waves")
            .and_then(|x| usize::try_from(x).ok())
            .unwrap_or(current.waves),
        threads: pick("threads")
            .and_then(|x| u32::try_from(x).ok())
            .unwrap_or(current.threads),
    };
    eprintln!(
        "quip-screen: tune keep_per_s={keep_per_s} reads={} sweeps={} waves={} threads={}",
        1u32 << next.reads_log2,
        next.sweeps,
        next.waves,
        next.threads
    );
    (next != current).then_some(next)
}

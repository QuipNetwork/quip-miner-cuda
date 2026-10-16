//! Collect nonce-aligned probe and production MSA minima on CUDA device 0.

use clap::Parser;
use quip_protocol::derive::NonceDeriver;
use quip_solver_core::adapt::adapt_params;
use quip_solver_core::{
    CancelToken, IsingGraph, SampleParams, StreamJob, StreamOutcome, StreamResult, TopologyView,
};
use quip_solver_cuda::capacity::{msa_max_reads, MSA_DEFAULT_NODES};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::nvml_gov::UtilGovernor;
use quip_solver_cuda::screen::{Screen, ScreenDomain, ScreenParams};
use quip_solver_cuda::streaming::{run_stream, stream_width};
use quip_solver_cuda::{cuda_msa_identity, KernelKind};
use serde::Deserialize;
use serde_json::{json, Value};
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::channel;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const SHAPES: [(u32, usize); 4] = [(2, 256), (2, 512), (2, 1024), (3, 512)];
const LAST_PROOF: [u8; 32] = [0x11; 32];
const MINER: [u8; 32] = [0x22; 32];
const BASE_SALT: [u8; 32] = [0x33; 32];
const SEEDS: [u64; 2] = [0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210];

#[derive(Parser)]
struct Args {
    /// Number of sequential lease salts to draw.
    #[arg(long, default_value_t = 4096)]
    n: usize,
    /// Directory for corpus-N.csv and its JSON sidecar.
    #[arg(
        long,
        default_value = "/zfspool/quip-mining/bench-evidence/lease-screen"
    )]
    out: PathBuf,
    /// Coordinator target for production adaptation. Default selects hardest difficulty.
    #[arg(long, default_value_t = i64::MIN, allow_hyphen_values = true)]
    target_milli: i64,
}

#[derive(Deserialize)]
struct Fixture {
    nodes: Vec<u32>,
    edges: Vec<(u32, u32)>,
    allowed_h_milli: Vec<i32>,
}

fn topology() -> Result<TopologyView> {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../tests/fixtures/aglais.spec.json"))?;
    TopologyView::from_proto(&quip_proto::v1::Topology {
        nodes: fixture.nodes,
        edges: Some(quip_proto::v1::EdgeList {
            u: fixture.edges.iter().map(|&(u, _)| u).collect(),
            v: fixture.edges.iter().map(|&(_, v)| v).collect(),
        }),
        allowed_h_milli: fixture.allowed_h_milli,
        allowed_j_milli: vec![-1000, 1000],
        ..Default::default()
    })
    .map_err(|error| format!("invalid fixture topology: {error:?}").into())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

fn nonces(count: usize) -> Result<Vec<[u8; 32]>> {
    let deriver = NonceDeriver::new(LAST_PROOF, MINER);
    (0..count)
        .map(|index| {
            let mut salt = BASE_SALT;
            salt[..8].copy_from_slice(&u64::try_from(index)?.to_le_bytes());
            Ok(deriver.derive(salt))
        })
        .collect()
}

fn probes(
    device: &CudaDevice,
    topo: &TopologyView,
    nonces: &[[u8; 32]],
) -> Result<(Vec<Vec<i64>>, Vec<Value>)> {
    let mut energies = Vec::new();
    let mut phases = Vec::new();
    for (reads_log2, sweeps) in SHAPES {
        let setup = Instant::now();
        let params = ScreenParams {
            reads_log2,
            sweeps,
            ..ScreenParams::default()
        };
        let screen = Screen::open(device, ScreenDomain::check(device, topo, params)?)?;
        let setup_seconds = setup.elapsed().as_secs_f64();
        // Warm uploads and JIT outside the timed corpus pass.
        let warmup = Instant::now();
        screen.run(&nonces[..nonces.len().min(32)], &|| false)?;
        let warmup_seconds = warmup.elapsed().as_secs_f64();
        let start = Instant::now();
        let result = screen.run(nonces, &|| false)?;
        let seconds = start.elapsed().as_secs_f64();
        if result.stopped || result.best.len() != nonces.len() {
            return Err("probe returned an incomplete corpus".into());
        }
        eprintln!("probe {reads_log2}/{sweeps}: {seconds:.3} s");
        energies.push(result.best);
        phases.push(json!({
            "reads_log2": reads_log2, "reads": 1u32 << reads_log2,
            "sweeps": sweeps, "threads": params.threads,
            "setup_seconds": setup_seconds, "warmup_seconds": warmup_seconds,
            "seconds": seconds,
        }));
    }
    Ok((energies, phases))
}

fn full_solve(
    device: &CudaDevice,
    topo: &TopologyView,
    nonces: &[[u8; 32]],
    params: &SampleParams,
) -> Result<Vec<i64>> {
    let width = stream_width(device, KernelKind::Msa);
    let (job_tx, job_rx) = channel::<StreamJob>(2 * width);
    let (result_tx, mut result_rx) = channel::<StreamResult>(2 * width);
    let edges: Arc<[(usize, usize)]> = topo.edges.as_slice().into();
    // Separate producer and consumer prevent bounded-channel backpressure deadlocks.
    std::thread::scope(|scope| {
        let producer = scope.spawn(move || -> Result<()> {
            for (index, nonce) in nonces.iter().enumerate() {
                let (h, j) = topo.draw(*nonce)?;
                let graph = IsingGraph {
                    h: h.into_iter().map(|v| f64::from(v) / 1000.0).collect(),
                    j: j.into_iter().map(|v| f64::from(v) / 1000.0).collect(),
                    edges: Arc::clone(&edges),
                };
                job_tx.blocking_send(StreamJob {
                    job_id: u64::try_from(index)?.to_le_bytes().to_vec(),
                    graph,
                    params: params.clone(),
                    watermark: None,
                })?;
            }
            Ok(())
        });
        let worker = scope.spawn(move || {
            let gov = UtilGovernor::start(&device.pci_bus_id, 100, false);
            run_stream(
                device,
                KernelKind::Msa,
                job_rx,
                result_tx,
                CancelToken::default(),
                &gov,
            );
        });
        let mut energies = vec![None; nonces.len()];
        let result = (|| -> Result<Vec<i64>> {
            let mut completed = 0;
            while let Some(result) = result_rx.blocking_recv() {
                let index = usize::try_from(u64::from_le_bytes(
                    result
                        .job_id
                        .try_into()
                        .map_err(|_| "invalid full-solve job id")?,
                ))?;
                let StreamOutcome::Completed(reads) = result.outcome else {
                    return Err("full solve was cancelled".into());
                };
                let reads = reads.map_err(|error| format!("full solve failed: {error:?}"))?;
                if reads.len() != params.num_reads {
                    return Err("full solve returned the wrong read count".into());
                }
                let energy = reads
                    .iter()
                    .map(|r| r.energy_milli)
                    .min()
                    .ok_or("full solve returned no reads")?;
                let slot = energies
                    .get_mut(index)
                    .ok_or("full-solve id outside corpus")?;
                if slot.replace(energy).is_some() {
                    return Err("duplicate full-solve result".into());
                }
                completed += 1;
                if completed % 256 == 0 || completed == nonces.len() {
                    eprintln!(
                        "full seed {:016x}: {completed}/{}",
                        params.seed,
                        nonces.len()
                    );
                }
            }
            energies
                .into_iter()
                .map(|v| v.ok_or_else(|| "missing full-solve result".into()))
                .collect()
        })();
        // A receiver error closes output before joining, allowing the driver to stop.
        drop(result_rx);
        worker.join().map_err(|_| "full-solve worker panicked")?;
        producer
            .join()
            .map_err(|_| "full-solve producer panicked")??;
        result
    })
}

fn revision() -> Result<String> {
    // A mounted worktree may have an inaccessible host-side .git pointer.
    if let Ok(commit) = std::env::var("CALIB_COMMIT") {
        return Ok(commit);
    }
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    if !output.status.success() {
        return Err("cannot read commit: set CALIB_COMMIT when mounting a worktree".into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.n == 0 {
        return Err("--n must be positive".into());
    }
    if std::env::var("CUDA_VISIBLE_DEVICES").as_deref() != Ok("0") {
        return Err("run with CUDA_VISIBLE_DEVICES=0 to isolate device 0".into());
    }
    let commit = revision()?;
    fs::create_dir_all(&args.out)?;
    let start = Instant::now();
    let topo = topology()?;
    let nonces = nonces(args.n)?;
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, MSA_DEFAULT_NODES)?;
    let reads = u32::try_from(msa_max_reads(device.msa_replica_words))?;
    let identity = cuda_msa_identity(device.max_nodes, reads);
    let adapted = adapt_params(
        args.target_milli,
        1,
        topo.num_nodes,
        topo.edges.len(),
        &topo.allowed_h_milli,
        &identity.adapt,
    );
    let mut params = SampleParams {
        num_reads: usize::try_from(adapted.num_reads)?,
        num_sweeps: usize::try_from(adapted.num_sweeps)?,
        ..SampleParams::default()
    };
    let setup_seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{}: {} reads, {} sweeps, {} replica words",
        device.name()?,
        params.num_reads,
        params.num_sweeps,
        device.msa_replica_words
    );
    let (probe, phases) = probes(&device, &topo, &nonces)?;
    let mut full = Vec::new();
    let mut full_seconds = Vec::new();
    for seed in SEEDS {
        params.seed = seed;
        let phase = Instant::now();
        full.push(full_solve(&device, &topo, &nonces, &params)?);
        full_seconds.push(phase.elapsed().as_secs_f64());
    }
    let path = args.out.join(format!("corpus-{}.csv", args.n));
    let mut csv = BufWriter::new(File::create(&path)?);
    writeln!(
        csv,
        "index,nonce,probe_2_256,probe_2_512,probe_2_1024,probe_3_512,full_a,full_b"
    )?;
    for (index, nonce) in nonces.iter().enumerate() {
        write!(csv, "{index},{}", hex(nonce))?;
        for column in &probe {
            write!(csv, ",{}", column[index])?;
        }
        writeln!(csv, ",{},{}", full[0][index], full[1][index])?;
    }
    csv.flush()?;
    let metadata = json!({
        "commit": commit, "n": args.n, "energy_units": "exact milli",
        "device": { "name": device.name()?, "index": 0, "pci_bus_id": device.pci_bus_id,
            "sms": device.max_sms, "max_nodes": device.max_nodes,
            "msa_replica_words": device.msa_replica_words },
        "lease": { "last_proof_block_hash": hex(&LAST_PROOF), "miner_account": hex(&MINER),
            "base_salt": hex(&BASE_SALT), "counter": "index in salt[0..8], little-endian, starting at 0" },
        "topology": { "nodes": topo.num_nodes, "edges": topo.edges.len(),
            "allowed_h_milli": topo.allowed_h_milli, "allowed_j_milli": topo.allowed_j_milli },
        "probe_shapes": phases,
        "full": { "reads": params.num_reads, "sweeps": params.num_sweeps,
            "sweeps_per_beta": params.sweeps_per_beta, "schedule": params.schedule.name(),
            "beta_range": params.beta_range, "target_milli": args.target_milli, "min_solutions": 1,
            "derivation": "cuda_msa_identity + core adapt_params, no overrides, default hardest target",
            "api": "streaming::run_stream", "seeds": SEEDS,
            "seed_note": "production streaming mixes the session seed with wall time; only nonce corpus is deterministic",
            "seconds": full_seconds },
        "setup_seconds": setup_seconds, "total_seconds": start.elapsed().as_secs_f64(),
        "timing_note": "consult external device-0 preflight and activity records before interpreting timings",
    });
    let mut sidecar = BufWriter::new(File::create(path.with_extension("json"))?);
    serde_json::to_writer_pretty(&mut sidecar, &metadata)?;
    writeln!(sidecar)?;
    sidecar.flush()?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

//! GPU-gated soak that drives the production streaming loop for hours.
//!
//! This is the vehicle for QUI-870's measurement and the evidence QUI-922's
//! verdict needs. It is deliberately an `#[ignore]`d test rather than a
//! throwaway script: the question ("does throughput decay with uptime?") recurs
//! every release, and a reviewable harness that lives next to the code beats
//! re-deriving one each time.
//!
//! Ignored by default — it needs a real CUDA device and runs for as long as you
//! tell it to. Typical use:
//!
//! ```text
//! QUIP_DRIVER_BUDGET=1 \
//! QUIP_DRIVER_BUDGET_WINDOW=300 \
//! QUIP_DRIVER_BUDGET_OUT=/tmp/soak_budget.jsonl \
//! QUIP_SOAK_MINUTES=240 \
//! QUIP_SOAK_SAMPLE_OUT=/tmp/soak_samples.csv \
//! cargo test --release --test driver_budget_soak -- --ignored --nocapture
//! ```
//!
//! Two independent records come out, and they are meant to be read together:
//!
//! * the driver budget JSONL — where the driver's wall clock went, and
//! * the sample CSV — `att/s` next to board power, core clock and temperature,
//!   read from NVML rather than from our own logs, so a decay here cannot be a
//!   log-pipeline artifact (the trap that made QUI-922 a follow-up rather than
//!   a fix).
//!
//! # Defaults
//!
//! The workload defaults mirror the QUI-922 report: 256 reads (that run pinned
//! `ADAPT_MIN_READS=256`) against an Advantage2-scale topology. Everything is
//! overridable so the light-workload regime from QUI-882 can be soaked too.

use quip_miner_core::{Algorithm, CancelGuard, IsingGraph, SampleParams, StreamJob};
use quip_miner_cuda::cuda_device::CudaDevice;
use quip_miner_cuda::nvml_gov::UtilGovernor;
use quip_miner_cuda::streaming::run_stream;
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::channel;

/// Read a `usize` knob from the environment, falling back to `default`.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic 64-bit LCG. Good enough to lay out a representative topology
/// and to jitter biases; nothing here depends on statistical quality, and a
/// fixed sequence keeps two soaks comparable.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        // Modulo bias is irrelevant for topology layout.
        usize::try_from(self.next_u64() >> 1).unwrap_or(0) % bound.max(1)
    }
}

/// Build a connected, Advantage2-scale Ising graph.
///
/// A ring guarantees connectivity, then random chords bring the average degree
/// up to `degree`. Biases and couplings stay in `{-1.0, 1.0}` so the int8
/// quantize the kernel applies is lossless and the soak measures the driver
/// rather than quantization behaviour.
fn build_graph(nodes: usize, degree: usize, seed: u64) -> IsingGraph {
    let mut rng = Lcg(seed);
    let mut edges: Vec<(usize, usize)> = (0..nodes).map(|i| (i, (i + 1) % nodes)).collect();

    let target = nodes * degree / 2;
    let mut placed: std::collections::HashSet<(usize, usize)> = edges.iter().copied().collect();
    while edges.len() < target {
        let a = rng.below(nodes);
        let b = rng.below(nodes);
        if a == b {
            continue;
        }
        let edge = if a < b { (a, b) } else { (b, a) };
        if placed.insert(edge) {
            edges.push(edge);
        }
    }

    let h = (0..nodes)
        .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    let j = (0..edges.len())
        .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    IsingGraph::new(h, j, edges)
}

/// Append one row and flush it.
///
/// Failures are reported rather than discarded: losing rows silently is how a
/// long soak ends with nothing to show for itself. Flushed per row so a soak
/// killed at hour three still leaves everything it measured on disk.
fn write_row(file: &mut std::fs::File, row: &str) {
    if let Err(e) = writeln!(file, "{row}").and_then(|()| file.flush()) {
        eprintln!("soak: sample row lost: {e}");
    }
}

/// One NVML reading, paired with the throughput observed since the last one.
struct Sample {
    uptime: Duration,
    att_per_s: f64,
    power_w: f64,
    clock_mhz: u32,
    temp_c: u32,
}

impl Sample {
    fn csv_row(&self) -> String {
        format!(
            "{:.1},{:.3},{:.1},{},{}",
            self.uptime.as_secs_f64() / 60.0,
            self.att_per_s,
            self.power_w,
            self.clock_mhz,
            self.temp_c,
        )
    }
}

/// Sample NVML until `stop`, recording `att/s` alongside board power.
///
/// Power is the load-bearing measurement: QUI-867 and QUI-922 both hinge on
/// power falling while the core clock stays pinned, which is what separates a
/// software stall from hardware throttling.
fn sample_loop(
    device_index: u32,
    completions: &AtomicU64,
    stop: &AtomicBool,
    out: Option<&str>,
    interval: Duration,
) {
    use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};

    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        eprintln!("soak: NVML unavailable; power/clock will not be recorded");
        return;
    };
    let Ok(dev) = nvml.device_by_index(device_index) else {
        eprintln!("soak: NVML device {device_index} unavailable");
        return;
    };

    let mut file = out.and_then(|path| match std::fs::File::create(path) {
        Ok(mut f) => {
            write_row(&mut f, "uptime_min,att_per_s,power_w,clock_mhz,temp_c");
            Some(f)
        }
        Err(e) => {
            eprintln!("soak: cannot write {path}: {e}");
            None
        }
    });

    let started = Instant::now();
    let mut last_count = 0u64;
    let mut last_at = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(250));
        if last_at.elapsed() < interval {
            continue;
        }
        let now_count = completions.load(Ordering::Relaxed);
        let elapsed = last_at.elapsed().as_secs_f64();
        // Completion counts in a 30s window are far inside f64's exact range.
        #[allow(clippy::cast_precision_loss)]
        let delta = (now_count - last_count) as f64;
        let sample = Sample {
            uptime: started.elapsed(),
            att_per_s: if elapsed > 0.0 { delta / elapsed } else { 0.0 },
            power_w: f64::from(dev.power_usage().unwrap_or(0)) / 1000.0,
            clock_mhz: dev.clock_info(Clock::Graphics).unwrap_or(0),
            temp_c: dev.temperature(TemperatureSensor::Gpu).unwrap_or(0),
        };
        println!(
            "[soak] up={:.1}min att/s={:.2} power={:.1}W clock={}MHz temp={}C",
            sample.uptime.as_secs_f64() / 60.0,
            sample.att_per_s,
            sample.power_w,
            sample.clock_mhz,
            sample.temp_c,
        );
        if let Some(f) = file.as_mut() {
            write_row(f, &sample.csv_row());
        }
        last_count = now_count;
        last_at = Instant::now();
    }
}

/// Drive `run_stream` under a steady job supply and record throughput against
/// board power for the configured duration.
#[test]
#[ignore = "requires a CUDA GPU and runs for QUIP_SOAK_MINUTES"]
fn stream_driver_soak() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let device_index = env_usize("QUIP_SOAK_DEVICE", 0);
    let minutes = env_usize("QUIP_SOAK_MINUTES", 5);
    let nodes = env_usize("QUIP_SOAK_NODES", 4577);
    let degree = env_usize("QUIP_SOAK_DEGREE", 18);
    let reads = env_usize("QUIP_SOAK_READS", 256);
    let sweeps = env_usize("QUIP_SOAK_SWEEPS", 1000);
    let sample_out = std::env::var("QUIP_SOAK_SAMPLE_OUT").ok();
    // Completions arrive as a burst of `stream_width` at a time, so a sampling
    // interval near one batch period aliases hard: the 30s default read a
    // steady 2.4 att/s as alternating 1.6 and 3.2. Long soaks want several
    // batches per sample.
    let sample_secs = env_usize("QUIP_SOAK_SAMPLE_SECS", 30);

    let device = CudaDevice::open(device_index).expect("open CUDA device");
    let graph = build_graph(nodes, degree, 0x5EED);
    println!(
        "[soak] {} min, {} nodes / {} edges, {reads} reads x {sweeps} sweeps on device {device_index}",
        minutes,
        graph.num_nodes(),
        graph.edges.len(),
    );

    let (job_tx, job_rx) = channel::<StreamJob>(64);
    let (res_tx, mut res_rx) = channel(64);
    let completions = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let deadline = Instant::now()
        + Duration::from_secs(u64::try_from(minutes).unwrap_or(5).saturating_mul(60));

    // Yielding off: this soak measures the driver alone on the card, and a
    // governor that could end sessions would confound the decay question.
    let gov = UtilGovernor::start(u32::try_from(device_index).unwrap_or(0), 100, false);
    let gov_ref = &gov;
    let graph_ref = &graph;
    let device_ref = &device;
    let completions_ref = &completions;
    let stop_ref = &stop;

    thread::scope(|s| {
        s.spawn(move || {
            run_stream(
                device_ref,
                Algorithm::Sa,
                job_rx,
                res_tx,
                CancelGuard::default(),
                gov_ref,
            );
        });

        // Feeder. The bounded channel is the backpressure: `blocking_send`
        // parks once the driver is 64 jobs behind, so the soak measures the
        // driver's ceiling rather than the feeder's.
        s.spawn(move || {
            let mut n = 0u64;
            while Instant::now() < deadline {
                let job = StreamJob {
                    job_id: n.to_le_bytes().to_vec(),
                    graph: graph_ref.clone(),
                    params: SampleParams {
                        num_reads: reads,
                        num_sweeps: sweeps,
                        sweeps_per_beta: 1,
                        beta_range: None,
                        seed: n.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    },
                    generation: 0,
                };
                if job_tx.blocking_send(job).is_err() {
                    break;
                }
                n += 1;
            }
            // Dropping the sender is what lets `run_stream` drain and return.
            drop(job_tx);
        });

        s.spawn(move || {
            sample_loop(
                u32::try_from(device_index).unwrap_or(0),
                completions_ref,
                stop_ref,
                sample_out.as_deref(),
                Duration::from_secs(u64::try_from(sample_secs).unwrap_or(30)),
            );
        });

        while res_rx.blocking_recv().is_some() {
            completions.fetch_add(1, Ordering::Relaxed);
        }
        stop.store(true, Ordering::Relaxed);
    });

    let total = completions.load(Ordering::Relaxed);
    println!("[soak] finished: {total} results");
    assert!(total > 0, "soak produced no results at all");
}

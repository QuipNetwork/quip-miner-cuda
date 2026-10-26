//! GPU-gated bench-path tests. `#[ignore]` so headless CI reports them
//! ignored rather than passed ([quip-miner-cuda-gp2] part b). Run on a CUDA
//! host: `cargo test -p quip-solver-cuda -- --ignored`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use quip_protocol::scoring::energy_milli;
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::streaming::bench_one;
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};
use serial_test::serial;
use std::sync::OnceLock;

fn device() -> &'static CudaDevice {
    static DEV: OnceLock<CudaDevice> = OnceLock::new();
    DEV.get_or_init(|| CudaDevice::open(0).expect("CUDA device 0 required for bench_live tests"))
}

fn ring(n: usize) -> IsingGraph {
    let h = vec![0.0; n];
    let j = vec![1.0; n];
    let edges = (0..n).map(|i| (i, (i + 1) % n)).collect();
    IsingGraph::new(h, j, edges)
}

#[test]
#[ignore = "requires a CUDA device"]
#[serial]
fn bench_one_scores_and_reports_positive_kernel_time() {
    let graph = ring(64);
    let params = SampleParams {
        num_reads: 8,
        num_sweeps: 1024,
        sweeps_per_beta: 4,
        ..SampleParams::default()
    };
    let (reads, timings) = bench_one(device(), &graph, &params, KernelKind::Sa, None).unwrap();
    assert_eq!(reads.len(), 8);
    for r in &reads {
        assert_eq!(
            r.energy_milli,
            energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
    assert!(
        timings.kernel_ns > 0,
        "event-measured kernel time must be positive"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
#[serial]
fn bench_one_kernel_time_grows_with_sweeps() {
    let graph = ring(64);
    let params = |num_sweeps| SampleParams {
        num_reads: 8,
        num_sweeps,
        sweeps_per_beta: 4,
        ..SampleParams::default()
    };
    let lo = params(1024);
    let hi = params(8192);
    let (_, t_lo) = bench_one(device(), &graph, &lo, KernelKind::Sa, None).unwrap();
    let (_, t_hi) = bench_one(device(), &graph, &hi, KernelKind::Sa, None).unwrap();
    // 8x the sweeps must cost meaningfully more device time (allow scheduler noise).
    assert!(
        t_hi.kernel_ns > t_lo.kernel_ns,
        "more sweeps must take longer"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
#[serial]
fn bench_one_upload_and_download_are_true_transfer_time_not_host_enqueue() {
    // The async-transfer caveat (module docs, plan Task 3): `upload_ns` /
    // `download_ns` are event-bracketed on `stream_transfer` and read after a
    // `synchronize()`, so they measure real device transfer time, not the
    // near-instant host enqueue of `memcpy_htod`/`clone_dtoh`. A larger graph
    // should therefore show a larger (not merely nonzero) transfer time.
    let small = ring(16);
    let large = ring(2048);
    let params = SampleParams {
        num_reads: 8,
        num_sweeps: 256,
        sweeps_per_beta: 4,
        ..SampleParams::default()
    };
    let (_, t_small) = bench_one(device(), &small, &params, KernelKind::Sa, None).unwrap();
    let (_, t_large) = bench_one(device(), &large, &params, KernelKind::Sa, None).unwrap();
    assert!(t_small.upload_ns > 0);
    assert!(t_large.upload_ns > 0);
    assert!(t_large.upload_ns >= t_small.upload_ns);
}

#[test]
#[ignore = "requires a CUDA device"]
#[serial]
fn bench_one_gibbs_scores_and_reports_positive_kernel_time() {
    let graph = ring(64);
    let params = SampleParams {
        num_reads: 8,
        num_sweeps: 1024,
        sweeps_per_beta: 4,
        ..SampleParams::default()
    };
    let (reads, timings) = bench_one(device(), &graph, &params, KernelKind::Gibbs, None).unwrap();
    assert_eq!(reads.len(), 8);
    for r in &reads {
        assert_eq!(
            r.energy_milli,
            energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
    assert!(timings.kernel_ns > 0);
}

/// msa through the isolated bench path: scored reads and a positive kernel
/// time, like SA and Gibbs.
#[test]
#[ignore = "requires a CUDA device"]
#[serial]
fn bench_one_msa_scores_and_reports_positive_kernel_time() {
    let graph = ring(64);
    let params = SampleParams {
        num_reads: 8,
        num_sweeps: 64,
        sweeps_per_beta: 4,
        seed: 7,
        ..Default::default()
    };
    let (reads, timings) = bench_one(device(), &graph, &params, KernelKind::Msa, None).unwrap();
    assert_eq!(reads.len(), 8);
    assert!(timings.kernel_ns > 0);
}

/// Nonce 0's reads must not depend on how many other nonces share the spill
/// buffer: a wrong per-nonce offset would let them overwrite it.
#[test]
#[ignore = "requires a CUDA device"]
#[serial]
fn nonce_zero_reads_ignore_concurrent_spilled_nonces() {
    use quip_solver_cuda::streaming::{bench_many, stream_width};
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 5000).unwrap();
    let graph = ring(3000);
    let params = SampleParams {
        num_reads: 128,
        num_sweeps: 32,
        sweeps_per_beta: 4,
        seed: 5,
        ..SampleParams::default()
    };
    let (alone, _) = bench_many(&device, &graph, &params, KernelKind::Msa, None, 1).unwrap();
    device.msa_shared_spins_cap = Some(64);
    let width = stream_width(&device, KernelKind::Msa);
    let (crowded, _) = bench_many(&device, &graph, &params, KernelKind::Msa, None, width).unwrap();
    let a: Vec<_> = alone.iter().map(|r| r.spins.clone()).collect();
    let c: Vec<_> = crowded.iter().map(|r| r.spins.clone()).collect();
    assert_eq!(a, c);
}

/// Real benchmark timing must keep host energies in the original f64 units.
#[test]
#[ignore = "requires a CUDA device"]
#[serial]
fn bench_real_coefficients_keep_real_energies_and_timing() {
    use quip_solver_cuda::streaming::bench_many_real;
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 64).unwrap();
    let graph = IsingGraph::new(
        vec![std::f64::consts::PI; 12],
        vec![-std::f64::consts::SQRT_2; 12],
        (0..12).map(|i| (i, (i + 1) % 12)).collect(),
    );
    let params = SampleParams {
        num_reads: 64,
        num_sweeps: 256,
        seed: 7,
        ..SampleParams::default()
    };
    let (reads, timing) = bench_many_real(&device, &graph, &params, 1).unwrap();
    assert_eq!(reads.len(), 64);
    assert!(timing.kernel_ns > 0);
    for read in reads {
        let fields: f64 = read
            .spins
            .iter()
            .map(|&s| std::f64::consts::PI * f64::from(s))
            .sum();
        let pairs: f64 = graph
            .edges
            .iter()
            .map(|&(u, v)| {
                -std::f64::consts::SQRT_2 * f64::from(read.spins[u]) * f64::from(read.spins[v])
            })
            .sum();
        assert!((read.energy - fields - pairs).abs() < 1e-10);
    }
}

/// The device read limit must agree with both the samples and emitted metadata.
#[test]
#[ignore = "requires a CUDA device"]
#[serial]
fn msa_bench_record_reads_match_returned_samples() {
    use quip_solver_cuda::capacity::msa_device_reads;
    use quip_solver_cuda::schema::BenchRecord;
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 64).unwrap();
    let num_reads = msa_device_reads(&device.limits);
    let params = SampleParams {
        num_reads,
        num_sweeps: 2,
        ..SampleParams::default()
    };
    let (reads, _) = bench_one(&device, &ring(64), &params, KernelKind::Msa, None).unwrap();
    let out = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_quip-cuda-msa"))
        .args([
            "bench",
            "run",
            "--reads",
            &num_reads.to_string(),
            "--sweeps",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--out",
        ])
        .arg(out.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let entries: Vec<_> = std::fs::read_dir(out.path()).unwrap().collect();
    assert_eq!(entries.len(), 1);
    let text = std::fs::read_to_string(entries[0].as_ref().unwrap().path()).unwrap();
    let record: BenchRecord = serde_json::from_str(text.trim()).unwrap();
    assert_eq!(record.reads, u64::try_from(reads.len()).unwrap());
    let scores = record
        .parts
        .iter()
        .find(|p| p.part == "energy_score")
        .unwrap();
    assert_eq!(scores.count, record.reads);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_quip-cuda-msa"))
        .args([
            "bench",
            "run",
            "--reads",
            &(num_reads + 1).to_string(),
            "--out",
        ])
        .arg(out.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("outside"));
}

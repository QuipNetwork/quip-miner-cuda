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

/// `n` nodes, each coupled with J = +1 to the next `k` around the ring. Every
/// local field is `2k`, so `k` up to 31 stays on SA's int8 path.
fn circulant(n: usize, k: usize) -> IsingGraph {
    let edges: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (1..=k).map(move |d| (i, (i + d) % n)))
        .collect();
    IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges)
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
    //
    // A 16-node and a 2048-node ring both upload a few KB at most, so their
    // gap sat inside per-copy latency and failed under a busy shared runner.
    // The large graph uploads about 250 KB of J instead, and each side keeps
    // its fastest run: contention only adds time, so the minimum is the one
    // closest to the true transfer cost.
    const RUNS: usize = 5;
    let small = ring(16);
    let large = circulant(4096, 31);
    let params = SampleParams {
        num_reads: 8,
        num_sweeps: 256,
        sweeps_per_beta: 4,
        ..SampleParams::default()
    };
    let fastest_upload = |graph: &IsingGraph| {
        (0..RUNS)
            .map(|_| {
                let (_, t) = bench_one(device(), graph, &params, KernelKind::Sa, None).unwrap();
                assert!(t.upload_ns > 0);
                t.upload_ns
            })
            .min()
            .unwrap()
    };
    let small_ns = fastest_upload(&small);
    let large_ns = fastest_upload(&large);
    assert!(
        large_ns > small_ns,
        "fastest upload: {large_ns} ns for the large graph, {small_ns} ns for the small one"
    );
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

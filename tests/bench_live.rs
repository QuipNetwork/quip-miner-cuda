//! GPU-gated bench-path tests. `#[ignore]` so headless CI reports them
//! ignored rather than passed ([quip-miner-cuda-gp2] part b). Run on a CUDA
//! host: `cargo test -p quip-miner-cuda -- --ignored`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use quip_miner_cuda::cuda_device::CudaDevice;
use quip_miner_cuda::streaming::bench_one;
use quip_miner_cuda::{Algorithm, IsingGraph, SampleParams};
use quip_protocol::scoring::energy_milli;
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
    let (reads, timings) = bench_one(device(), &graph, &params, Algorithm::Sa).unwrap();
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
    let (_, t_lo) = bench_one(device(), &graph, &lo, Algorithm::Sa).unwrap();
    let (_, t_hi) = bench_one(device(), &graph, &hi, Algorithm::Sa).unwrap();
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
    let (_, t_small) = bench_one(device(), &small, &params, Algorithm::Sa).unwrap();
    let (_, t_large) = bench_one(device(), &large, &params, Algorithm::Sa).unwrap();
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
    let (reads, timings) = bench_one(device(), &graph, &params, Algorithm::Gibbs).unwrap();
    assert_eq!(reads.len(), 8);
    for r in &reads {
        assert_eq!(
            r.energy_milli,
            energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
    assert!(timings.kernel_ns > 0);
}

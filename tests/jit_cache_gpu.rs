//! GPU-gated round-trip for the NVRTC PTX cache (`src/jit_cache.rs`).
//!
//! Ignored by default: it needs a real CUDA device. Run with
//! `cargo test --release --test jit_cache_gpu -- --ignored`.

use quip_miner_cuda::cuda_device::CudaDevice;
use std::time::Instant;

#[test]
#[ignore = "requires a CUDA GPU"]
fn second_open_uses_cached_ptx() {
    // Isolate the cache in a per-process temp dir so the run is hermetic.
    let dir = std::env::temp_dir().join(format!("quip-jitcache-test-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::env::set_var("QUIP_CUDA_CACHE", &dir);
    std::env::remove_var("QUIP_CUDA_CACHE_DISABLE");

    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    // Cold open: compiles both kernels via NVRTC and writes the cache.
    let t0 = Instant::now();
    let _cold_dev = CudaDevice::open(0).expect("first open");
    let cold = t0.elapsed();

    // One PTX file per kernel (sa + gibbs) must exist after the first open.
    let ptx: Vec<_> = std::fs::read_dir(&dir)
        .expect("cache dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "ptx"))
        .collect();
    assert_eq!(ptx.len(), 2, "expected sa + gibbs cached PTX, got {ptx:?}");

    // Warm open: same process, same dir — loads the cached PTX, no NVRTC.
    let t1 = Instant::now();
    let _warm_dev = CudaDevice::open(0).expect("second open");
    let warm = t1.elapsed();

    eprintln!("cold open {cold:?}, warm open {warm:?}");
    assert!(
        warm < cold,
        "warm open ({warm:?}) should beat cold NVRTC compile ({cold:?})"
    );

    drop(std::fs::remove_dir_all(&dir));
}

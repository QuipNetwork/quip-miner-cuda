//! GPU-gated checks that a resolved node capacity reaches the kernel.
//!
//! Ignored by default: these need a real CUDA device. Run with
//! `cargo test --release --test node_capacity_gpu -- --ignored`.
//!
//! Every bound asserted here was measured on an NVIDIA RTX A4000. See
//! `docs/superpowers/specs/2026-08-06-configurable-node-capacity-design.md`
//! for the sweep those numbers come from.

use quip_miner_cuda::capacity::{GIBBS_DEFAULT_NODES, SA_MAX_NODES};
use quip_miner_cuda::cuda_device::CudaDevice;
use quip_miner_cuda::Algorithm;

/// Pegasus P16 has 5640 nodes, above both shipped defaults (5000 / 4800).
/// Both kernels must compile and open there, which is the whole point of the
/// change.
#[test]
#[ignore = "requires a CUDA GPU"]
fn both_algorithms_open_at_pegasus_scale() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let sa = CudaDevice::open_with_nodes(0, Algorithm::Sa, 5640).expect("SA at 5640");
    assert_eq!(sa.max_nodes, 5640);

    let gibbs = CudaDevice::open_with_nodes(0, Algorithm::Gibbs, 5640).expect("Gibbs at 5640");
    assert_eq!(gibbs.max_nodes, 5640);
}

/// Gibbs holds its state in shared memory and measured flat cost per node out
/// to 48000, so a capacity far above the default must still open.
#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_opens_well_above_the_shared_default() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let dev = CudaDevice::open_with_nodes(0, Algorithm::Gibbs, 32768).expect("Gibbs at 32768");
    assert_eq!(dev.max_nodes, 32768);
}

/// SA fails with `CUDA_ERROR_ILLEGAL_ADDRESS` at 16384, which the array size
/// alone does not explain. Until that defect is found, the open must refuse
/// rather than reach the range that corrupts memory.
#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_refuses_above_its_bound() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let err = CudaDevice::open_with_nodes(0, Algorithm::Sa, 16384)
        .expect_err("SA above its bound must fail at open");
    let msg = err.to_string();
    assert!(
        msg.contains("16384"),
        "message must name the request: {msg}"
    );
    assert!(
        msg.contains(&SA_MAX_NODES.to_string()),
        "message must name the limit: {msg}"
    );
}

/// The Gibbs ceiling comes from the device, not a constant. On a 48 KB
/// shared-memory part this refuses 65536 and names the budget.
#[test]
#[ignore = "requires a CUDA GPU"]
fn gibbs_refuses_above_the_device_budget() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let err = CudaDevice::open_with_nodes(0, Algorithm::Gibbs, 65536)
        .expect_err("Gibbs above the device budget must fail at open");
    let msg = err.to_string();
    assert!(
        msg.contains("65536"),
        "message must name the request: {msg}"
    );
    assert!(
        msg.contains("budget"),
        "message must name the device budget: {msg}"
    );
}

/// Two capacities in one process must both load the kernel they asked for. A
/// cache key that ignored capacity would serve the first PTX to the second
/// open, and the kernel would write past a state array sized for the smaller
/// run with no error.
#[test]
#[ignore = "requires a CUDA GPU"]
fn two_capacities_in_one_process_do_not_cross_serve() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let small = CudaDevice::open_with_nodes(0, Algorithm::Gibbs, GIBBS_DEFAULT_NODES)
        .expect("Gibbs at the default");
    assert_eq!(small.max_nodes, GIBBS_DEFAULT_NODES);

    let large = CudaDevice::open_with_nodes(0, Algorithm::Gibbs, 16384).expect("Gibbs at 16384");
    assert_eq!(large.max_nodes, 16384);
}

//! GPU-gated checks that a resolved node capacity reaches the kernel.
//!
//! Ignored by default: these need a real CUDA device. Run with
//! `cargo test --release --test node_capacity_gpu -- --ignored`.
//!
//! Every bound asserted here was measured on an NVIDIA RTX A4000. See
//! `docs/superpowers/specs/2026-08-06-configurable-node-capacity-design.md`
//! for the sweep those numbers come from.

use quip_miner_cuda::capacity::GIBBS_DEFAULT_NODES;
use quip_miner_cuda::capacity::{msa_max_reads, MSA_DEFAULT_NODES, MSA_REPLICA_WORDS};
use quip_miner_cuda::cuda_device::{probe_msa_replica_words, CudaDevice};
use quip_miner_cuda::sampler::{sample_ising, SampleError};
use quip_miner_cuda::streaming::max_reads;
use quip_miner_cuda::{IsingGraph, KernelKind, SampleParams};

/// Pegasus P16 has 5640 nodes, above both shipped defaults (5000 / 4800).
/// Both kernels must compile and open there, which is the whole point of the
/// change.
#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_and_gibbs_open_at_pegasus_scale() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let sa = CudaDevice::open_with_nodes(0, KernelKind::Sa, 5640).expect("SA at 5640");
    assert_eq!(sa.max_nodes, 5640);

    let gibbs = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 5640).expect("Gibbs at 5640");
    assert_eq!(gibbs.max_nodes, 5640);
}

/// msa holds its spin state in dynamic shared memory. An A4000 opts in to
/// 101376 bytes, which `capacity::msa_budget` puts at 5791 spins for two
/// replica words, so Pegasus P16 fits. The open also refuses a loaded
/// kernel whose static shared size exceeds `MSA_STATIC_SHARED_BYTES`.
#[test]
#[ignore = "requires a CUDA GPU"]
fn msa_opens_at_pegasus_scale() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let dev = CudaDevice::open_with_nodes(0, KernelKind::Msa, 5640).expect("msa at 5640");
    assert_eq!(dev.max_nodes, 5640);
}

/// The msa ceiling comes from the opt-in shared memory, not a constant, so
/// this refuses 65536 and names the shared-memory budget.
#[test]
#[ignore = "requires a CUDA GPU"]
fn msa_refuses_above_the_shared_memory_budget() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let err = CudaDevice::open_with_nodes(0, KernelKind::Msa, 65536)
        .expect_err("msa above the shared-memory budget must fail at open");
    let msg = err.to_string();
    assert!(
        msg.contains("65536"),
        "message must name the request: {msg}"
    );
    assert!(
        msg.contains("shared-memory budget"),
        "message must name the resource that bound it: {msg}"
    );
}

/// The msa kernel unrolls `capacity::MSA_MAX_DEGREE` neighbours per
/// spin. A denser graph is refused per job as `Unsupported`, which the
/// wire maps to `Capacity`, and the session stays usable.
#[test]
#[ignore = "requires a CUDA GPU"]
fn msa_refuses_a_spin_with_more_than_twenty_neighbours() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, MSA_DEFAULT_NODES)
        .expect("msa opens at its default");
    // A star: node 0 has 21 neighbours, one above the budget.
    let edges: Vec<(usize, usize)> = (1..=21).map(|i| (0, i)).collect();
    let graph = IsingGraph::new(vec![0.0; 22], vec![1.0; 21], edges);
    let params = SampleParams {
        num_reads: 8,
        num_sweeps: 16,
        seed: 1,
        ..Default::default()
    };
    let Err(SampleError::Unsupported(msg)) =
        sample_ising(&device, &graph, &params, KernelKind::Msa)
    else {
        panic!("a degree-21 graph must be refused as Unsupported");
    };
    assert!(
        msg.contains("max degree 21") && msg.contains("20-neighbour budget"),
        "message must name the degree and the budget: {msg}"
    );
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

    let dev = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 32768).expect("Gibbs at 32768");
    assert_eq!(dev.max_nodes, 32768);
}

/// SA is bounded by device memory, which no card satisfies at this size, so
/// the refusal must name the request and the memory budget rather than let
/// the driver fail later with a bare out-of-memory.
#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_refuses_above_the_device_memory_budget() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let err = CudaDevice::open_with_nodes(0, KernelKind::Sa, 10_000_000)
        .expect_err("SA above the device memory budget must fail at open");
    let msg = err.to_string();
    assert!(
        msg.contains("10000000"),
        "message must name the request: {msg}"
    );
    assert!(
        msg.contains("memory budget"),
        "message must name the resource that bound it: {msg}"
    );
}

/// SA now reaches far past its old 8192 cap. 16384 was the original
/// `CUDA_ERROR_ILLEGAL_ADDRESS` repro and must open cleanly.
#[test]
#[ignore = "requires a CUDA GPU"]
fn sa_opens_at_the_original_illegal_address_repro() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let dev = CudaDevice::open_with_nodes(0, KernelKind::Sa, 16384).expect("SA at 16384");
    assert_eq!(dev.max_nodes, 16384);
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

    let err = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 65536)
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

    let small = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, GIBBS_DEFAULT_NODES)
        .expect("Gibbs at the default");
    assert_eq!(small.max_nodes, GIBBS_DEFAULT_NODES);

    let large = CudaDevice::open_with_nodes(0, KernelKind::Gibbs, 16384).expect("Gibbs at 16384");
    assert_eq!(large.max_nodes, 16384);
}

/// The read count a miner declares comes from `probe_msa_replica_words`, which
/// reads one device attribute before any kernel compiles. The count the
/// session then launches with comes from the opened device. The two derive the
/// same number from the same attribute, so a drift between them would make the
/// miner advertise reads it clamps away (quip-miner-cuda-9p5).
#[test]
#[ignore = "requires a CUDA GPU"]
fn the_probed_replica_words_match_the_opened_device() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }

    let probed = probe_msa_replica_words(0, MSA_DEFAULT_NODES)
        .expect("a device that opens msa holds at least one replica word");
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, MSA_DEFAULT_NODES)
        .expect("msa opens at its default");
    assert_eq!(
        probed, device.msa_replica_words,
        "probed replica words must match the opened device"
    );
    assert!(
        (1..=MSA_REPLICA_WORDS).contains(&probed),
        "replica words {probed} outside 1..={MSA_REPLICA_WORDS}"
    );
    // Reads are 64 per word, and the declared cap must equal the launch cap.
    assert_eq!(
        max_reads(KernelKind::Msa, probed),
        u32::try_from(msa_max_reads(device.msa_replica_words)).expect("small"),
    );
}

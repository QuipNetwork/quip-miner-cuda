//! Spilled msa spin state on a real GPU. Spill placement moves spin words
//! between shared and global memory without touching the RNG stream or the
//! update order, so every cap must return the all-shared samples bit for bit.
//! `#[ignore]`: run with `GPU_TEST` (see the plan) or `make test-gpu`.

use quip_solver_cuda::capacity::MSA_DEFAULT_NODES;
use quip_solver_cuda::cuda_device::{CudaDevice, MsaLane};
use quip_solver_cuda::sampler::sample_ising;
use quip_solver_cuda::streaming::{bench_one, msa_layout};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams, SamplerResult};
use serial_test::serial;

/// Circulant graph: node i links to i+1 ..= i+degree/2 (mod n), so every
/// node has exactly `degree` neighbours when n > degree. Couplings take
/// both signs and fields take all of -1, 0, +1, so the field bond and both
/// coupling signs are exercised.
#[allow(clippy::cast_precision_loss)] // small indices
fn signed_circulant(n: usize, degree: usize) -> IsingGraph {
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for i in 0..n {
        for d in 1..=degree / 2 {
            edges.push((i, (i + d) % n));
            j.push(if (i * 7 + d) % 3 == 0 { -1.0 } else { 1.0 });
        }
    }
    let h = (0..n).map(|i| (i % 3) as f64 - 1.0).collect();
    IsingGraph::new(h, j, edges)
}

fn params() -> SampleParams {
    SampleParams {
        num_reads: 128,
        num_sweeps: 64,
        sweeps_per_beta: 4,
        seed: 7,
        ..SampleParams::default()
    }
}

fn spins(reads: &[SamplerResult]) -> Vec<Vec<i8>> {
    reads.iter().map(|r| r.spins.clone()).collect()
}

fn open() -> CudaDevice {
    CudaDevice::open_with_nodes(0, KernelKind::Msa, MSA_DEFAULT_NODES).expect("msa opens")
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn unit_spill_matches_all_shared_at_every_cap() {
    let mut device = open();
    let n = 1000;
    let graph = signed_circulant(n, 20);
    let shared = spins(&sample_ising(&device, &graph, &params(), KernelKind::Msa).unwrap());
    for cap in [0, 1, n - 1, n, n + 5] {
        device.msa_shared_spins_cap = Some(cap);
        assert_eq!(msa_layout(&device, n, 128).shared_spins, cap.min(n));
        let spilled = sample_ising(&device, &graph, &params(), KernelKind::Msa).unwrap();
        assert_eq!(spins(&spilled), shared, "unit samples differ at cap {cap}");
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn weighted_spill_matches_all_shared_on_both_lanes() {
    let mut device = open();
    let n = 600;
    let graph = signed_circulant(n, 20);
    for lane in [MsaLane::I32, MsaLane::F32] {
        device.msa_shared_spins_cap = None;
        let (shared, _) =
            bench_one(&device, &graph, &params(), KernelKind::Msa, Some(lane)).unwrap();
        for cap in [0, 1, n - 1, n] {
            device.msa_shared_spins_cap = Some(cap);
            assert_eq!(msa_layout(&device, n, 128).shared_spins, cap.min(n));
            let (spilled, _) =
                bench_one(&device, &graph, &params(), KernelKind::Msa, Some(lane)).unwrap();
            assert_eq!(
                spins(&spilled),
                spins(&shared),
                "{lane:?} differs at cap {cap}"
            );
        }
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn degree_40_spill_matches_all_shared() {
    let mut device = open();
    let n = 800;
    let graph = signed_circulant(n, 40);
    let shared = spins(&sample_ising(&device, &graph, &params(), KernelKind::Msa).unwrap());
    for cap in [0, n / 2] {
        device.msa_shared_spins_cap = Some(cap);
        let spilled = sample_ising(&device, &graph, &params(), KernelKind::Msa).unwrap();
        assert_eq!(
            spins(&spilled),
            shared,
            "degree-40 samples differ at cap {cap}"
        );
    }
    for lane in [MsaLane::I32, MsaLane::F32] {
        device.msa_shared_spins_cap = None;
        let (all, _) = bench_one(&device, &graph, &params(), KernelKind::Msa, Some(lane)).unwrap();
        device.msa_shared_spins_cap = Some(0);
        let (spilled, _) =
            bench_one(&device, &graph, &params(), KernelKind::Msa, Some(lane)).unwrap();
        assert_eq!(spins(&spilled), spins(&all), "{lane:?} degree 40");
    }
}

/// Historical spill fixture: 200,000 nodes at degree 40. Every read
/// must come back, and the host-scored energy must equal consensus scoring.
#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn samples_200k_nodes_at_degree_40() {
    use quip_protocol::scoring::energy_milli;
    let device =
        CudaDevice::open_with_nodes(0, KernelKind::Msa, 200_000).expect("msa at 200k nodes");
    let graph = signed_circulant(200_000, 40);
    assert_eq!(graph.edges.len(), 4_000_000);
    let p = SampleParams {
        num_reads: 128,
        num_sweeps: 16,
        sweeps_per_beta: 4,
        seed: 11,
        ..SampleParams::default()
    };
    let reads = sample_ising(&device, &graph, &p, KernelKind::Msa)
        .expect("sample the historical spill fixture");
    assert_eq!(reads.len(), 128);
    for r in reads.iter().take(2) {
        assert_eq!(
            r.energy_milli,
            energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
}

/// Above the shared budget with no cap: spill happens naturally, and it still
/// matches a run with a much smaller shared part.
#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn natural_spill_matches_a_smaller_shared_part() {
    let mut device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 20_000).expect("msa at 20000");
    let graph = signed_circulant(20_000, 20);
    let natural = spins(&sample_ising(&device, &graph, &params(), KernelKind::Msa).unwrap());
    device.msa_shared_spins_cap = Some(1000);
    let small = spins(&sample_ising(&device, &graph, &params(), KernelKind::Msa).unwrap());
    assert_eq!(small, natural);
}

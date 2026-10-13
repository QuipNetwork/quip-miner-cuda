//! Consensus parity: host `energy_milli` matches golden vectors, and live GPU
//! samples score bit-exactly to consensus `energy_milli`. Live reads are also
//! a pure function of the whole 64-bit seed.
//!
//! Host/golden tests always run. Live sampling tests are `#[ignore]` so default
//! `cargo test` on headless CI reports them as ignored, not silently passed
//! ([quip-miner-cuda-gp2] part b). Run on a GPU host:
//! `cargo test -p quip-solver-cuda -- --ignored`.

// Test fixtures parse golden-vector literals; casts are intentional and values are call-site visible.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use quip_protocol::scoring::energy_milli;
use quip_protocol::wire::{decode_spins, encode_spins};
use quip_solver_conformance::GOLDEN_VECTORS;
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::sampler::sample_ising;
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};
use serde_json::Value;
use serial_test::serial;
use std::sync::OnceLock;

fn golden() -> Value {
    serde_json::from_str(GOLDEN_VECTORS).unwrap()
}

fn device(kind: KernelKind) -> &'static CudaDevice {
    static SA: OnceLock<CudaDevice> = OnceLock::new();
    static MSA: OnceLock<CudaDevice> = OnceLock::new();
    static GIBBS: OnceLock<CudaDevice> = OnceLock::new();
    let cell = match kind {
        KernelKind::Sa => &SA,
        KernelKind::Msa => &MSA,
        KernelKind::Gibbs => &GIBBS,
    };
    cell.get_or_init(|| {
        CudaDevice::open_with_nodes(0, kind, 0).unwrap_or_else(|e| {
            panic!("CUDA device 0 required for golden_parity tests: {e}");
        })
    })
}

/// Golden Ising cases: host consensus scorer matches the golden energy, and a
/// wire round-trip preserves it. No GPU needed (the sampler always scores
/// host-side with `energy_milli` for consensus).
#[test]
fn host_energy_matches_golden_vectors() {
    for case in golden()["energy"].as_array().unwrap() {
        let spins: Vec<i8> = case["spins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect();
        let h: Vec<f64> = case["h_milli"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as f64 / 1000.0)
            .collect();
        let j: Vec<f64> = case["j_milli"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as f64 / 1000.0)
            .collect();
        let edges: Vec<(usize, usize)> = case["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e[0].as_u64().unwrap() as usize,
                    e[1].as_u64().unwrap() as usize,
                )
            })
            .collect();

        let expected = case["energy_milli"].as_i64().unwrap();
        assert_eq!(
            energy_milli(&spins, &h, &j, &edges),
            expected,
            "host energy_milli mismatch"
        );

        // Wire round-trip must preserve scoring.
        let bytes = encode_spins(&spins);
        let decoded = decode_spins(&bytes).unwrap();
        assert_eq!(energy_milli(&decoded, &h, &j, &edges), expected);
    }
}

/// Truncation cases from golden vectors (toward zero, not round-to-nearest).
#[test]
fn truncation_matches_golden() {
    for case in golden()["truncation"].as_array().unwrap() {
        let energy = case["energy"].as_f64().unwrap();
        assert_eq!(
            (energy * 1000.0) as i64,
            case["energy_milli"].as_i64().unwrap()
        );
    }
}

/// Live SA, msa and Gibbs sample: every returned energy equals consensus scoring.
#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
fn live_sample_energies_match_energy_milli() {
    let graph = IsingGraph::new(
        vec![1.0, -0.5, 0.0, 0.25],
        vec![1.0, -1.0, -1.0, 1.0],
        vec![(0, 1), (1, 2), (2, 3), (0, 3)],
    );
    let params = SampleParams {
        num_reads: 16,
        num_sweeps: 64,
        seed: 12345,
        ..Default::default()
    };

    for kernel in [KernelKind::Sa, KernelKind::Msa, KernelKind::Gibbs] {
        let graph = &graph;
        let results = sample_ising(device(kernel), graph, &params, kernel).expect("sample");
        assert_eq!(results.len(), 16);
        for r in &results {
            let expected = energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges);
            assert_eq!(
                r.energy_milli, expected,
                "{kernel:?} reported energy_milli {} != consensus {}",
                r.energy_milli, expected
            );
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
            let bytes = encode_spins(&r.spins);
            assert_eq!(decode_spins(&bytes).unwrap(), r.spins);
        }
    }
}

/// SA finds ferro ground state (sanity that the kernel actually anneals).
#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
fn sa_finds_ground_state_on_ferro() {
    let dev = device(KernelKind::Sa);
    let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
    let params = SampleParams {
        num_reads: 16,
        num_sweeps: 128,
        seed: 42,
        ..Default::default()
    };
    let results = sample_ising(dev, &graph, &params, KernelKind::Sa).expect("sa");
    assert!(
        results.iter().any(|r| r.energy_milli == -1000),
        "SA failed to find ferro ground: {:?}",
        results.iter().map(|r| r.energy_milli).collect::<Vec<_>>()
    );
}

/// msa finds the ground state of a ferro pair under positive fields. The
/// untouched all-`+1` state scores `+1000`, so the assertion fails unless the
/// kernel anneals; the ground state `(-1, -1)` scores `-3000`.
#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
fn msa_finds_ground_state_on_ferro_with_fields() {
    let dev = device(KernelKind::Msa);
    let graph = IsingGraph::new(vec![1.0, 1.0], vec![-1.0], vec![(0, 1)]);
    let params = SampleParams {
        num_reads: 16,
        num_sweeps: 128,
        seed: 42,
        ..Default::default()
    };
    let results = sample_ising(dev, &graph, &params, KernelKind::Msa).expect("msa");
    assert!(
        results.iter().any(|r| r.energy_milli == -3000),
        "msa failed to find the ferro ground state under fields: {:?}",
        results.iter().map(|r| r.energy_milli).collect::<Vec<_>>()
    );
}

/// The same seed gives the same reads on every kernel, and a seed that differs
/// only in bit 63 gives different ones. The kernels' RNG state is 32-bit, and
/// keeping the seed's low half made the second pair identical (QUI-1627).
///
/// The anneal is hot and short, so each of the 64 spins in a read is close to
/// a fair coin: the two seeds' 16 reads agreeing by chance is about 2^-1024.
/// The ring's couplings are ±1 and its degree is 2, inside msa's limits.
#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
fn reads_repeat_for_a_seed_and_change_with_its_high_bit() {
    let n = 64;
    let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    let j: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let graph = IsingGraph::new(vec![0.0; n], j, edges);
    let params = |seed: u64| SampleParams {
        num_reads: 16,
        num_sweeps: 4,
        beta_range: Some((0.01, 0.02)),
        seed,
        ..Default::default()
    };
    let seed = 0x0123_4567_89ab_cdef_u64;

    for kernel in [KernelKind::Sa, KernelKind::Msa, KernelKind::Gibbs] {
        let dev = device(kernel);
        let first = sample_ising(dev, &graph, &params(seed), kernel).expect("sample");
        let again = sample_ising(dev, &graph, &params(seed), kernel).expect("sample");
        let flipped = sample_ising(dev, &graph, &params(seed ^ (1 << 63)), kernel).expect("sample");
        assert_eq!(first.len(), 16);
        assert_eq!(
            first, again,
            "{kernel:?}: the same seed gave different reads"
        );
        assert_ne!(
            first, flipped,
            "{kernel:?}: seeds differing only in bit 63 gave the same reads"
        );
    }
}

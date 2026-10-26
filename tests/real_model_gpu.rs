//! Precision storage and real-valued MSA sampling on a CUDA device.

use quip_solver_cuda::capacity::{msa_device_reads, MsaStorage};
use quip_solver_cuda::coefficients::{plan, CoefficientPath};
use quip_solver_cuda::cuda_device::{CudaDevice, MsaLane};
use quip_solver_cuda::sampler::SampleError;
use quip_solver_cuda::streaming::{bench_many_real, bench_one, sample_one, sample_one_real};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};

fn device() -> CudaDevice {
    CudaDevice::open_with_nodes(0, KernelKind::Msa, 50_000).unwrap()
}

fn params() -> SampleParams {
    SampleParams {
        num_reads: 128,
        num_sweeps: 4096,
        sweeps_per_beta: 4,
        seed: 17,
        ..SampleParams::default()
    }
}

fn weighted_graph(n: usize) -> IsingGraph {
    IsingGraph::new(
        (0..n)
            .map(|i| if i % 2 == 0 { 0.016 } else { -0.992 })
            .collect(),
        (0..n)
            .map(|i| if i % 2 == 0 { -0.976 } else { 0.032 })
            .collect(),
        (0..n).map(|i| (i, (i + 1) % n)).collect(),
    )
}

fn energy(spins: &[i8], graph: &IsingGraph) -> f64 {
    let mut result = 0.0;
    for (&h, &spin) in graph.h.iter().zip(spins) {
        result += h * f64::from(spin);
    }
    for (&j, &(u, v)) in graph.j.iter().zip(graph.edges.iter()) {
        result += j * f64::from(spins[u]) * f64::from(spins[v]);
    }
    result
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn exact_milli_graph_ignores_float_path() {
    let device = device();
    let params = SampleParams {
        num_sweeps: 16,
        beta_range: Some((0.01, 0.1)),
        ..params()
    };
    for graph in [
        weighted_graph(12),
        IsingGraph::new(
            vec![0.0; 12],
            vec![-1.0; 12],
            (0..12).map(|i| (i, (i + 1) % 12)).collect(),
        ),
    ] {
        let exact = sample_one(&device, &graph, &params, KernelKind::Msa).unwrap();
        let real = sample_one_real(&device, &graph, &params).unwrap();
        assert_eq!(exact.len(), msa_device_reads(&device.limits));
        assert_eq!(real.len(), msa_device_reads(&device.limits));
        for (a, b) in exact.iter().zip(&real) {
            assert_eq!(a.spins, b.spins);
            assert!((b.energy - energy(&b.spins, &graph)).abs() < 1e-12);
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn real_sampling_refuses_f32_beta_overflow() {
    let device = device();
    let graph = IsingGraph::new(vec![1.2e-38; 64], vec![], vec![]);
    let error = sample_one_real(&device, &graph, &params()).unwrap_err();
    let SampleError::Unsupported(message) = error else {
        panic!("expected unsupported beta range, got {error:?}");
    };
    assert!(message.contains("beta"), "{message}");
    assert!(message.contains("f32"), "{message}");
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn real_bench_refuses_f32_beta_overflow() {
    let device = device();
    let graph = IsingGraph::new(vec![1.2e-38; 64], vec![], vec![]);
    let error = bench_many_real(&device, &graph, &params(), 1).unwrap_err();
    let SampleError::Unsupported(message) = error else {
        panic!("expected unsupported beta range, got {error:?}");
    };
    assert!(message.contains("beta"), "{message}");
    assert!(message.contains("f32"), "{message}");
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn weighted_int8_matches_int32() {
    let mut device = device();
    device.msa_shared_spins_cap = Some(4);
    // Cover both the single-pass and chunked weighted hot loops, with spill.
    for degree in [2, 22, 62] {
        let n = 64;
        let mut graph = weighted_graph(n);
        graph.edges = (0..n)
            .flat_map(|i| (1..=degree / 2).map(move |d| (i, (i + d) % n)))
            .collect::<Vec<_>>()
            .into();
        graph.j = (0..graph.edges.len())
            .map(|i| if i % 2 == 0 { 0.032 } else { -0.976 })
            .collect();
        assert_eq!(
            plan(KernelKind::Msa, &graph, device.max_nodes)
                .unwrap()
                .path,
            CoefficientPath::WeightedInt8
        );
        let (narrow, _) = bench_one(&device, &graph, &params(), KernelKind::Msa, None).unwrap();
        let (wide, _) = bench_one(
            &device,
            &graph,
            &params(),
            KernelKind::Msa,
            Some(MsaLane::I32),
        )
        .unwrap();
        assert_eq!(
            narrow.iter().map(|r| &r.spins).collect::<Vec<_>>(),
            wide.iter().map(|r| &r.spins).collect::<Vec<_>>(),
            "degree {degree}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn off_grid_graph_runs_on_f32() {
    let device = device();
    let graph = IsingGraph::new(
        vec![std::f64::consts::SQRT_2 / 10.0; 12],
        (0..12)
            .map(|i| {
                if i % 2 == 0 {
                    -std::f64::consts::PI
                } else {
                    std::f64::consts::SQRT_2
                }
            })
            .collect(),
        (0..12).map(|i| (i, (i + 1) % 12)).collect(),
    );
    assert!(sample_one(&device, &graph, &params(), KernelKind::Msa).is_err());
    let ground = (0_u32..1 << 12)
        .map(|bits| {
            let spins: Vec<i8> = (0..12)
                .map(|i| if bits & (1 << i) == 0 { 1 } else { -1 })
                .collect();
            energy(&spins, &graph)
        })
        .fold(f64::INFINITY, f64::min);
    let reads = sample_one_real(&device, &graph, &params()).unwrap();
    assert_eq!(reads.len(), msa_device_reads(&device.limits));
    assert!(reads
        .iter()
        .any(|r| (r.energy - ground).abs() <= 1e-4 * ground.abs().max(1.0)));
    for read in reads {
        let recomputed = energy(&read.spins, &graph);
        assert!((read.energy - recomputed).abs() <= 1e-4 * recomputed.abs().max(1.0));
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn real_chunked_fields_keep_fractional_couplings_with_spill() {
    let mut device = device();
    device.msa_shared_spins_cap = Some(4);
    let nodes = 24;
    let edges: Vec<_> = (0..nodes)
        .flat_map(|u| (u + 1..nodes).map(move |v| (u, v)))
        .collect();
    let graph = IsingGraph::new(
        vec![std::f64::consts::PI / 100.0; nodes],
        vec![-std::f64::consts::SQRT_2 / 10.0; edges.len()],
        edges,
    );
    let ground = energy(&vec![-1; nodes], &graph);
    for num_reads in [64, 128]
        .into_iter()
        .filter(|&reads| reads <= msa_device_reads(&device.limits))
    {
        let params = SampleParams {
            num_reads,
            ..params()
        };
        let reads = sample_one_real(&device, &graph, &params).unwrap();
        assert_eq!(reads.len(), num_reads);
        assert!(reads
            .iter()
            .any(|r| (r.energy - ground).abs() < 1e-4 * ground.abs()));
    }
}

fn acceptance_oracle() -> Vec<Vec<i8>> {
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    let base_seed = mix(42) >> 32;
    assert_eq!(base_seed, 0xbdd7_3226);
    let slot_seed = mix(base_seed << 8);
    assert_eq!(slot_seed, 0xfbf9_46e1_d36c_5141);
    let p = (-2.0 * (2.0 * f64::from(0.125_000_01_f32))).exp();
    let oracle: Vec<Vec<i8>> = (0..128_u64)
        .map(|r| {
            (0..64_u64)
                .map(|node| {
                    let event = mix(slot_seed ^ (1 << 40) ^ node);
                    let uniform = (f64::from(u32::try_from(mix(event ^ r) >> 41).unwrap()) + 0.5)
                        / 8_388_608.0;
                    if uniform < p {
                        1
                    } else {
                        -1
                    }
                })
                .collect()
        })
        .collect();
    oracle
}

/// Pin independent acceptance draws across lanes, replica words and spill layouts.
#[test]
#[ignore = "requires a CUDA GPU"]
fn real_sparse_acceptance_matches_replica_oracle() {
    use quip_solver_core::BetaSchedule;
    use quip_solver_cuda::streaming::last_storage;

    fn mask(samples: &[Vec<i8>], node: usize) -> u64 {
        samples.iter().enumerate().fold(0, |bits, (r, spins)| {
            bits | (u64::from(spins[node] == 1) << r)
        })
    }

    let mut device = device();
    let max_reads = msa_device_reads(&device.limits);
    if max_reads == 64 {
        eprintln!(
            "63/64 replica word crossing requires a 128-read device; running 63 and 64 reads"
        );
    }
    let oracle = acceptance_oracle();
    let expected_masks = [
        [0xf993_8ace_786f_bd13, 0xad3e_4a59_54b9_5e76],
        [0xf5ac_4257_3db9_7b93, 0x97dd_dffb_57db_ed9c],
    ];
    for (node, expected) in expected_masks.iter().enumerate() {
        assert_eq!(mask(&oracle[..64], node), expected[0]);
        assert_eq!(mask(&oracle[64..], node), expected[1]);
    }
    assert_ne!(oracle[63][0], oracle[64][0]);
    let mut prefix: Option<Vec<Vec<i8>>> = None;
    for degree in [20, 22] {
        let edges: Vec<_> = (0..64)
            .flat_map(|node| (1..=degree / 2).map(move |d| (node, (node + d) % 64)))
            .collect();
        let graph = IsingGraph::new(vec![0.125_000_01_f64; 64], vec![0.0; edges.len()], edges);
        assert!(plan(KernelKind::Msa, &graph, device.max_nodes).is_err());
        for cap in [None, Some(0)] {
            device.msa_shared_spins_cap = cap;
            for num_reads in [63, 64, 65, 128].into_iter().filter(|&r| r <= max_reads) {
                let params = SampleParams {
                    num_reads,
                    num_sweeps: 2,
                    sweeps_per_beta: 1,
                    seed: 42,
                    schedule: BetaSchedule::Linear,
                    beta_range: Some((1_000_000.0, 2.0)),
                };
                let mut repeated = Vec::new();
                for _ in 0..2 {
                    let reads = sample_one_real(&device, &graph, &params).unwrap();
                    assert_eq!(last_storage(&device), Some(MsaStorage::Sparse));
                    assert_eq!(reads.len(), num_reads);
                    repeated.push(reads.into_iter().map(|read| read.spins).collect::<Vec<_>>());
                }
                let samples = &repeated[0];
                if num_reads >= 64 {
                    eprintln!("degree={degree} cap={cap:?} reads={num_reads} node0={:#018x} node1={:#018x}", mask(&samples[..64], 0), mask(&samples[..64], 1));
                }
                if num_reads == 128 {
                    eprintln!(
                        "second word: node0={:#018x} node1={:#018x}",
                        mask(&samples[64..], 0),
                        mask(&samples[64..], 1)
                    );
                }
                assert_eq!(
                    repeated[0], repeated[1],
                    "repeatability: degree={degree} cap={cap:?} reads={num_reads}"
                );
                if let Some(previous) = &prefix {
                    let common = previous.len().min(num_reads);
                    assert_eq!(
                        samples[..common],
                        previous[..common],
                        "prefix: degree={degree} cap={cap:?} reads={num_reads}"
                    );
                }
                prefix = Some(samples.clone());
                assert_eq!(
                    *samples,
                    oracle[..num_reads],
                    "oracle: degree={degree} cap={cap:?} reads={num_reads}"
                );
                if num_reads >= 64 {
                    for (node, expected) in expected_masks.iter().enumerate() {
                        assert_eq!(mask(&samples[..64], node), expected[0]);
                        if num_reads == 128 {
                            assert_eq!(mask(&samples[64..], node), expected[1]);
                        }
                    }
                }
            }
        }
    }
}

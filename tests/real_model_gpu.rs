//! Precision storage and real-valued MSA sampling on a CUDA device.

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
        assert_eq!(exact.len(), params.num_reads);
        assert_eq!(real.len(), params.num_reads);
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
    assert_eq!(reads.len(), 128);
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
    for num_reads in [64, 128] {
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

//! Dense MSA correctness and cancellation on a real GPU.

use quip_solver_core::{CancelToken, StreamJob, StreamOutcome, StreamResult};
use quip_solver_cuda::coefficients::{self, CoefficientPath};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::nvml_gov::UtilGovernor;
use quip_solver_cuda::streaming::{run_stream, sample_one, sample_one_real};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};
use serial_test::serial;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::channel;

fn sk(n: usize, mut seed: u64) -> IsingGraph {
    let mut edges = Vec::with_capacity(n * (n - 1) / 2);
    let mut j = Vec::with_capacity(edges.capacity());
    for u in 0..n {
        for v in u + 1..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            edges.push((u, v));
            j.push(if seed & 1 == 0 { -1.0 } else { 1.0 });
        }
    }
    IsingGraph::new(vec![0.0; n], j, edges)
}

fn open() -> CudaDevice {
    CudaDevice::open_with_nodes(0, KernelKind::Msa, 8_000)
        .unwrap()
        .with_dense_msa_for_test()
}

fn params(sweeps: usize) -> SampleParams {
    SampleParams {
        num_reads: 128,
        num_sweeps: sweeps,
        sweeps_per_beta: (sweeps / 64).max(1),
        seed: 42,
        ..SampleParams::default()
    }
}

// Gray-code enumeration updates the host energy after exactly one spin flip.
// The independent host scorer checks both the starting point and the minimum.
fn ground_energy(graph: &IsingGraph) -> i64 {
    let n = graph.num_nodes();
    let mut spins = vec![1i8; n];
    let mut energy = quip_protocol::scoring::energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
    let mut best = energy;
    let mut best_spins = spins.clone();
    for step in 1usize..1 << n {
        let node = step.trailing_zeros() as usize;
        let mut field = coefficients::to_milli(graph.h[node]).unwrap();
        for (&(u, v), &j) in graph.edges.iter().zip(&graph.j) {
            let coef = coefficients::to_milli(j).unwrap();
            if u == node {
                field += coef * i64::from(spins[v]);
            }
            if v == node {
                field += coef * i64::from(spins[u]);
            }
        }
        energy -= 2 * i64::from(spins[node]) * field;
        spins[node] = -spins[node];
        if energy < best {
            best = energy;
            best_spins.clone_from(&spins);
        }
    }
    assert_eq!(
        best,
        quip_protocol::scoring::energy_milli(&best_spins, &graph.h, &graph.j, &graph.edges)
    );
    best
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_unit_finds_brute_force_ground_state() {
    let device = open();
    for seed in 1..=10 {
        let graph = sk(20, seed);
        let ground = ground_energy(&graph);
        let reads = sample_one(&device, &graph, &params(4_096), KernelKind::Msa).unwrap();
        assert_eq!(reads.len(), 128);
        assert_eq!(
            reads.iter().map(|r| r.energy_milli).min(),
            Some(ground),
            "SK seed {seed}"
        );
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_energies_match_host() {
    let device = open();
    let mut graph = sk(500, 99);
    // Exercise both nonzero field signs as well as zero fields.
    for (i, h) in graph.h.iter_mut().enumerate() {
        *h = match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        };
    }
    let reads = sample_one(&device, &graph, &params(64), KernelKind::Msa).unwrap();
    assert_eq!(reads.len(), 128);
    for read in reads {
        assert_eq!(read.spins.len(), 500);
        assert!(read.spins.iter().all(|s| *s == -1 || *s == 1));
        assert_eq!(
            read.energy_milli,
            quip_protocol::scoring::energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_seed_is_deterministic() {
    let device = open();
    let graph = sk(100, 9);
    for count in [1, 64, 65, 128] {
        let mut p = params(64);
        p.num_reads = count;
        let a = sample_one(&device, &graph, &p, KernelKind::Msa).unwrap();
        let b = sample_one(&device, &graph, &p, KernelKind::Msa).unwrap();
        assert_eq!(a.len(), count);
        assert_eq!(
            a.iter().map(|r| &r.spins).collect::<Vec<_>>(),
            b.iter().map(|r| &r.spins).collect::<Vec<_>>()
        );
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_cancel_aborts() {
    let device = open();
    let graph = sk(4_000, 17);
    let gov = UtilGovernor::start(&device.pci_bus_id, 100, false);
    let cancel = CancelToken::default();
    let (job_tx, job_rx) = channel::<StreamJob>(4);
    let (res_tx, mut res_rx) = channel::<StreamResult>(4);
    std::thread::scope(|s| {
        let driver_cancel = cancel.clone();
        let device_ref = &device;
        let gov_ref = &gov;
        s.spawn(move || {
            run_stream(
                device_ref,
                KernelKind::Msa,
                job_rx,
                res_tx,
                driver_cancel,
                gov_ref,
            );
        });
        job_tx
            .blocking_send(StreamJob {
                job_id: vec![1],
                graph: graph.clone(),
                params: params(20_000_000),
                watermark: Some(1),
            })
            .unwrap();
        // A timed channel wait lets the driver upload and start the long model.
        let (_settle_tx, settle_rx) = std::sync::mpsc::channel::<()>();
        assert!(settle_rx.recv_timeout(Duration::from_secs(2)).is_err());
        let at = Instant::now();
        cancel.cancel_through(1);
        job_tx
            .blocking_send(StreamJob {
                job_id: vec![2],
                graph,
                params: params(1),
                watermark: Some(2),
            })
            .unwrap();
        drop(job_tx);
        let mut cancelled = false;
        let mut completed = false;
        while let Some(result) = res_rx.blocking_recv() {
            match result.job_id[0] {
                1 => {
                    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
                    assert!(at.elapsed() < Duration::from_secs(30));
                    cancelled = true;
                }
                2 => {
                    let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
                        panic!("next session did not complete");
                    };
                    assert_eq!(reads.len(), 128);
                    completed = true;
                }
                _ => panic!("unexpected job"),
            }
        }
        assert!(cancelled && completed);
    });
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_stream_rotates_slots_with_distinct_matrices() {
    let device = open();
    let n = 65usize;
    let total = device.max_sms * 3 + 1;
    let graph = sk(n, 31);
    let gov = UtilGovernor::start(&device.pci_bus_id, 100, false);
    let (job_tx, job_rx) = channel::<StreamJob>(total);
    let (res_tx, mut res_rx) = channel::<StreamResult>(total);
    for id in 0..total {
        let mut model = graph.clone();
        model.j.fill(if id % 2 == 0 { -1.0 } else { 1.0 });
        job_tx
            .blocking_send(StreamJob {
                job_id: id.to_le_bytes().to_vec(),
                graph: model,
                params: params(64),
                watermark: Some(1),
            })
            .unwrap();
    }
    drop(job_tx);
    run_stream(
        &device,
        KernelKind::Msa,
        job_rx,
        res_tx,
        CancelToken::default(),
        &gov,
    );
    let mut seen = vec![false; total];
    while let Some(result) = res_rx.blocking_recv() {
        let id = usize::from_le_bytes(result.job_id.try_into().unwrap());
        assert!(!seen[id]);
        seen[id] = true;
        let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
            panic!("slot job failed");
        };
        let ground = if id % 2 == 0 { -2_080_000 } else { -32_000 };
        assert_eq!(reads.len(), 128);
        assert_eq!(
            reads.iter().map(|r| r.energy_milli).min(),
            Some(ground),
            "slot job {id}"
        );
    }
    assert!(seen.into_iter().all(|v| v));
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_masks_absent_edges_and_constant_diagonal() {
    let device = open();
    let mut h: Vec<f64> = (0..65)
        .map(|i| match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();
    h[0] = 1.0;
    let graph = IsingGraph::new(h, vec![1.0], vec![(0, 0)]);
    let mut p = params(1);
    p.beta_range = Some((20.0, 20.0));
    let reads = sample_one(&device, &graph, &p, KernelKind::Msa).unwrap();
    assert_eq!(reads.len(), 128);
    // 43 nonzero fields contribute -43. The diagonal adds the constant +1.
    for read in reads {
        assert_eq!(read.energy_milli, -42_000);
        assert_eq!(read.spins[0], -1);
    }
}

fn random(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn weighted_graph(n: usize, mut seed: u64, path: CoefficientPath) -> IsingGraph {
    let mut coefficient = || match path {
        CoefficientPath::WeightedInt8 => {
            f64::from(i32::try_from(random(&mut seed) % 255).unwrap() - 127)
        }
        CoefficientPath::Wide => {
            f64::from(i32::try_from(random(&mut seed) % 2001).unwrap() - 1000) / 1000.0
        }
        CoefficientPath::Float32 => {
            // Box-Muller with open-interval uniforms, without a new dependency.
            let u = (f64::from(u32::try_from(random(&mut seed) >> 32).unwrap()) + 0.5)
                / 4_294_967_296.0;
            let v = (f64::from(u32::try_from(random(&mut seed) >> 32).unwrap()) + 0.5)
                / 4_294_967_296.0;
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
        CoefficientPath::Int8 => unreachable!(),
    };
    let h = (0..n).map(|_| coefficient()).collect();
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            edges.push((u, v));
            j.push(coefficient());
        }
    }
    let graph = IsingGraph::new(h, j, edges);
    if path != CoefficientPath::Float32 {
        assert_eq!(
            coefficients::plan(KernelKind::Msa, &graph, n).unwrap().path,
            path
        );
    }
    graph
}

fn host_real_energy(spins: &[i8], graph: &IsingGraph) -> f64 {
    let fields: f64 = graph
        .h
        .iter()
        .zip(spins)
        .map(|(h, s)| h * f64::from(*s))
        .sum();
    fields
        + graph
            .edges
            .iter()
            .zip(&graph.j)
            .map(|(&(u, v), j)| j * f64::from(spins[u]) * f64::from(spins[v]))
            .sum::<f64>()
}

fn ground_real_energy(graph: &IsingGraph) -> f64 {
    let n = graph.num_nodes();
    let mut spins = vec![1i8; n];
    let mut energy = host_real_energy(&spins, graph);
    let mut best = energy;
    let mut best_spins = spins.clone();
    for step in 1usize..1 << n {
        let node = step.trailing_zeros() as usize;
        let mut field = graph.h[node];
        for (&(u, v), &j) in graph.edges.iter().zip(&graph.j) {
            if u == node {
                field += j * f64::from(spins[v]);
            }
            if v == node {
                field += j * f64::from(spins[u]);
            }
        }
        energy -= 2.0 * f64::from(spins[node]) * field;
        spins[node] = -spins[node];
        if energy < best {
            best = energy;
            best_spins.clone_from(&spins);
        }
    }
    let scored = host_real_energy(&best_spins, graph);
    assert!((scored - best).abs() < 1e-6);
    scored
}

fn weighted_ground_states(path: CoefficientPath) {
    let device = open();
    for seed in 1..=10 {
        let graph = weighted_graph(20, seed, path);
        if path == CoefficientPath::Float32 {
            let ground = ground_real_energy(&graph);
            let reads = sample_one_real(&device, &graph, &params(4_096)).unwrap();
            assert_eq!(reads.len(), 128);
            let best = reads.iter().map(|r| r.energy).fold(f64::INFINITY, f64::min);
            assert!(
                (best - ground).abs() <= 1e-4 * ground.abs(),
                "seed {seed}: {best} != {ground}"
            );
        } else {
            let ground = ground_energy(&graph);
            let reads = sample_one(&device, &graph, &params(4_096), KernelKind::Msa).unwrap();
            assert_eq!(reads.len(), 128);
            assert_eq!(
                reads.iter().map(|r| r.energy_milli).min(),
                Some(ground),
                "seed {seed}"
            );
        }
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_int8_finds_brute_force_ground_state() {
    weighted_ground_states(CoefficientPath::WeightedInt8);
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_i32_finds_brute_force_ground_state() {
    weighted_ground_states(CoefficientPath::Wide);
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_f32_real_finds_brute_force_ground_state() {
    weighted_ground_states(CoefficientPath::Float32);
}

fn check_weighted_energies(device: &CudaDevice, graph: &IsingGraph, path: CoefficientPath) {
    if path == CoefficientPath::Float32 {
        let reads = sample_one_real(device, graph, &params(64)).unwrap();
        assert_eq!(reads.len(), 128);
        for read in reads {
            assert_eq!(read.spins.len(), graph.num_nodes());
            assert!(read.spins.iter().all(|&s| s == -1 || s == 1));
            assert!((read.energy - host_real_energy(&read.spins, graph)).abs() < 1e-8);
        }
    } else {
        let reads = sample_one(device, graph, &params(64), KernelKind::Msa).unwrap();
        assert_eq!(reads.len(), 128);
        for read in reads {
            assert_eq!(read.spins.len(), graph.num_nodes());
            assert!(read.spins.iter().all(|&s| s == -1 || s == 1));
            assert_eq!(
                read.energy_milli,
                quip_protocol::scoring::energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_weighted_energies_match_host() {
    let device = open();
    for path in [
        CoefficientPath::WeightedInt8,
        CoefficientPath::Wide,
        CoefficientPath::Float32,
    ] {
        check_weighted_energies(&device, &weighted_graph(500, 99, path), path);
    }
}

#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn dense_and_sparse_weighted_energies_match_host() {
    let sparse = CudaDevice::open_with_nodes(0, KernelKind::Msa, 500).unwrap();
    let dense = open();
    for path in [
        CoefficientPath::WeightedInt8,
        CoefficientPath::Wide,
        CoefficientPath::Float32,
    ] {
        let graph = weighted_graph(20, 99, path);
        check_weighted_energies(&sparse, &graph, path);
        check_weighted_energies(&dense, &graph, path);
    }
}

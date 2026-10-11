//! QUI-78 arbitrary-model cases on a real GPU. `#[ignore]`: run with
//! `make test-gpu`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // small literal fixtures

use quip_protocol::scoring::energy_milli;
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::sampler::{sample_ising, SampleError};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};
use serial_test::serial;
use std::sync::OnceLock;

fn device(kind: KernelKind) -> &'static CudaDevice {
    static SA: OnceLock<CudaDevice> = OnceLock::new();
    static GIBBS: OnceLock<CudaDevice> = OnceLock::new();
    static MSA: OnceLock<CudaDevice> = OnceLock::new();
    let cell = match kind {
        KernelKind::Sa => &SA,
        KernelKind::Gibbs => &GIBBS,
        KernelKind::Msa => &MSA,
    };
    cell.get_or_init(|| {
        CudaDevice::open_with_nodes(0, kind, 5000).unwrap_or_else(|e| panic!("CUDA device 0: {e}"))
    })
}

/// Lowest energy over all 2^n states, in milli. n <= 16.
fn brute_force_min(g: &IsingGraph) -> i64 {
    let n = g.h.len();
    (0u32..(1 << n))
        .map(|bits| {
            let spins: Vec<i8> = (0..n)
                .map(|i| if bits >> i & 1 == 1 { 1 } else { -1 })
                .collect();
            energy_milli(&spins, &g.h, &g.j, &g.edges)
        })
        .min()
        .unwrap()
}

/// Deterministic quarter-unit ring: h, J in {-0.75, -0.5, -0.25, 0.25, 0.5, 0.75}.
fn quarter_ring(n: usize) -> IsingGraph {
    let vals = [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75];
    let h = (0..n).map(|i| vals[(i * 7 + 3) % 6]).collect();
    let edges: Vec<_> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    let j = (0..n).map(|i| vals[(i * 5 + 1) % 6]).collect();
    IsingGraph::new(h, j, edges)
}

fn params() -> SampleParams {
    SampleParams {
        num_reads: 64,
        num_sweeps: 1024,
        sweeps_per_beta: 4,
        ..SampleParams::default()
    }
}

fn assert_finds_ground_state(kind: KernelKind, g: &IsingGraph) {
    let reads = sample_ising(device(kind), g, &params(), kind).expect("solves");
    for r in &reads {
        assert_eq!(
            r.energy_milli,
            energy_milli(&r.spins, &g.h, &g.j, &g.edges),
            "reported != recomputed"
        );
    }
    let best = reads.iter().map(|r| r.energy_milli).min().unwrap();
    assert_eq!(best, brute_force_min(g), "{kind:?} missed the ground state");
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn quarter_unit_qubo_reaches_brute_force_minimum() {
    let g = quarter_ring(12);
    assert_finds_ground_state(KernelKind::Sa, &g);
    assert_finds_ground_state(KernelKind::Gibbs, &g);
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn one_x_and_thousand_x_reach_the_same_ground_state() {
    let one = quarter_ring(12);
    let thousand = IsingGraph::new(
        one.h.iter().map(|v| v * 1000.0).collect(),
        one.j.iter().map(|v| v * 1000.0).collect(),
        one.edges.to_vec(),
    );
    assert_finds_ground_state(KernelKind::Sa, &one);
    assert_finds_ground_state(KernelKind::Sa, &thousand);
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn non_representable_models_are_refused() {
    let sub_milli = IsingGraph::new(vec![0.0005, 0.0], vec![1.0], vec![(0, 1)]);
    let weighted = IsingGraph::new(vec![1.0, 0.0], vec![2.0], vec![(0, 1)]);
    let wide_gibbs = IsingGraph::new(vec![128.0, 0.0], vec![1.0], vec![(0, 1)]);
    for (kind, g) in [
        (KernelKind::Sa, &sub_milli),
        (KernelKind::Gibbs, &sub_milli),
        (KernelKind::Msa, &weighted),
        (KernelKind::Gibbs, &wide_gibbs),
    ] {
        let err = sample_ising(device(kind), g, &params(), kind).unwrap_err();
        assert!(
            matches!(err, SampleError::Unsupported(_)),
            "{kind:?}: {err}"
        );
    }
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn stream_separates_one_x_and_thousand_x_schedules() {
    use quip_solver_core::{CancelToken, StreamJob, StreamOutcome, StreamResult};
    use quip_solver_cuda::nvml_gov::UtilGovernor;
    use quip_solver_cuda::streaming::run_stream;
    use tokio::sync::mpsc::channel;

    let one = quarter_ring(12);
    let thousand = IsingGraph::new(
        one.h.iter().map(|v| v * 1000.0).collect(),
        one.j.iter().map(|v| v * 1000.0).collect(),
        one.edges.to_vec(),
    );
    let graphs = [one, thousand];
    let (job_tx, job_rx) = channel::<StreamJob>(16);
    let (res_tx, mut res_rx) = channel::<StreamResult>(16);
    for (id, graph) in graphs.iter().enumerate() {
        job_tx
            .try_send(StreamJob {
                job_id: vec![u8::try_from(id).unwrap()],
                graph: graph.clone(),
                params: params(),
                watermark: None,
            })
            .unwrap();
    }
    drop(job_tx);
    let driver = std::thread::spawn(move || {
        let device = device(KernelKind::Sa);
        let gov = UtilGovernor::start(&device.pci_bus_id, 90, false);
        run_stream(
            device,
            KernelKind::Sa,
            job_rx,
            res_tx,
            CancelToken::default(),
            &gov,
        );
    });
    let mut seen = [false; 2];
    while let Some(result) = res_rx.blocking_recv() {
        let id = usize::from(result.job_id[0]);
        assert!(!seen[id], "duplicate result for job {id}");
        seen[id] = true;
        let StreamOutcome::Completed(reads) = result.outcome else {
            panic!("job {id} was cancelled");
        };
        let reads = reads.expect("stream solves");
        for read in &reads {
            let graph = &graphs[id];
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges),
                "reported != recomputed"
            );
        }
        let best = reads.iter().map(|read| read.energy_milli).min().unwrap();
        assert_eq!(
            best,
            brute_force_min(&graphs[id]),
            "job {id} missed the ground state"
        );
    }
    driver.join().expect("stream driver exits");
    assert_eq!(seen, [true, true], "both scales must produce a result");
}

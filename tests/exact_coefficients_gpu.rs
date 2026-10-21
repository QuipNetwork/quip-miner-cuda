//! QUI-78 arbitrary-model cases on a real GPU. `#[ignore]`: run with
//! `make test-gpu`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // small literal fixtures

use quip_protocol::scoring::energy_milli;
use quip_solver_cuda::cuda_device::{CudaDevice, MsaLane};
use quip_solver_cuda::sampler::{sample_ising, SampleError};
use quip_solver_cuda::streaming::{bench_one, max_reads};
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
    assert_ground_state(kind, g, &reads);
}

fn assert_ground_state(
    kind: KernelKind,
    g: &IsingGraph,
    reads: &[quip_solver_cuda::SamplerResult],
) {
    for r in reads {
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
    assert_finds_ground_state(KernelKind::Msa, &g);
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
    assert_finds_ground_state(KernelKind::Msa, &one);
    assert_finds_ground_state(KernelKind::Msa, &thousand);
}

fn assert_msa_weighted_ground_states(lane: MsaLane) {
    let one = quarter_ring(12);
    let thousand = IsingGraph::new(
        one.h.iter().map(|v| v * 1000.0).collect(),
        one.j.iter().map(|v| v * 1000.0).collect(),
        one.edges.to_vec(),
    );
    for graph in [&one, &thousand] {
        let (reads, _) = bench_one(
            device(KernelKind::Msa),
            graph,
            &params(),
            KernelKind::Msa,
            Some(lane),
        )
        .unwrap_or_else(|e| panic!("{lane:?}: {e}"));
        assert_ground_state(KernelKind::Msa, graph, &reads);
    }
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn msa_weighted_i32_reaches_quarter_and_scaled_ground_states() {
    assert_msa_weighted_ground_states(MsaLane::I32);
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn msa_weighted_f32_reaches_quarter_and_scaled_ground_states() {
    assert_msa_weighted_ground_states(MsaLane::F32);
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn msa_forced_weighted_lanes_solve_unit_models_at_supported_read_counts() {
    let dev = device(KernelKind::Msa);
    let read_cap = max_reads(KernelKind::Msa, dev.msa_replica_words);
    let graph = IsingGraph::new(
        vec![0.0; 12],
        vec![1.0; 12],
        (0..12).map(|i| (i, (i + 1) % 12)).collect(),
    );
    for lane in [MsaLane::I32, MsaLane::F32] {
        for num_reads in [64, 128] {
            if num_reads == 128 && read_cap < 128 {
                continue;
            }
            let params = SampleParams {
                num_reads,
                ..params()
            };
            let (reads, _) = bench_one(dev, &graph, &params, KernelKind::Msa, Some(lane))
                .expect("forced weighted unit model");
            assert_eq!(reads.len(), num_reads);
            assert_ground_state(KernelKind::Msa, &graph, &reads);
        }
    }
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn weighted_msa_requires_an_msa_device_open() {
    let err = sample_ising(
        device(KernelKind::Sa),
        &quarter_ring(12),
        &params(),
        KernelKind::Msa,
    )
    .unwrap_err();
    assert!(matches!(err, SampleError::Unsupported(_)), "{err}");
    assert!(
        err.to_string().contains("requires an MSA device open"),
        "{err}"
    );
}

/// Ferromagnetic `K_n` with every field -1: all spins +1 is the unique ground
/// state, at -(n(n-1)/2) * `scale_j` - n * `scale_h`.
fn ferro_complete(n: usize, scale_j: f64, scale_h: f64) -> IsingGraph {
    let mut edges = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            edges.push((a, b));
        }
    }
    let j = vec![-scale_j; edges.len()];
    IsingGraph::new(vec![-scale_h; n], j, edges)
}

fn ferro_ground_milli(n: usize, scale_j: i64, scale_h: i64) -> i64 {
    let n = i64::try_from(n).unwrap();
    -1000 * (n * (n - 1) / 2 * scale_j + n * scale_h)
}

/// Degree 21 is the first two-pass spin and degree 62 the ceiling. Every
/// lane must reach the ground state, which a wrong carry in the chunked
/// counter or the chunked weighted field would miss.
#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn msa_reaches_ground_states_at_degree_21_and_62_on_every_lane() {
    let p = SampleParams {
        num_reads: 64,
        num_sweeps: 512,
        seed: 3,
        ..SampleParams::default()
    };
    for n in [22, 63] {
        let unit = ferro_complete(n, 1.0, 1.0);
        let (reads, _) =
            bench_one(device(KernelKind::Msa), &unit, &p, KernelKind::Msa, None).unwrap();
        let best = reads.iter().map(|r| r.energy_milli).min().unwrap();
        assert_eq!(best, ferro_ground_milli(n, 1, 1), "unit K_{n}");

        let weighted = ferro_complete(n, 2.0, 3.0);
        for lane in [MsaLane::I32, MsaLane::F32] {
            let (reads, _) = bench_one(
                device(KernelKind::Msa),
                &weighted,
                &p,
                KernelKind::Msa,
                Some(lane),
            )
            .unwrap();
            let best = reads.iter().map(|r| r.energy_milli).min().unwrap();
            assert_eq!(best, ferro_ground_milli(n, 2, 3), "{lane:?} K_{n}");
        }
    }
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn msa_refuses_k64_on_unit_and_weighted_paths() {
    let graph = complete_pm1(64);
    for lane in [None, Some(MsaLane::I32), Some(MsaLane::F32)] {
        let err = bench_one(
            device(KernelKind::Msa),
            &graph,
            &params(),
            KernelKind::Msa,
            lane,
        )
        .unwrap_err();
        assert!(matches!(err, SampleError::Unsupported(_)), "{err}");
        assert!(
            err.to_string().contains("63") && err.to_string().contains("62-neighbour"),
            "{err}"
        );
    }
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn non_representable_models_are_refused() {
    let sub_milli = IsingGraph::new(vec![0.0005, 0.0], vec![1.0], vec![(0, 1)]);
    let wide_gibbs = IsingGraph::new(vec![128.0, 0.0], vec![1.0], vec![(0, 1)]);
    for (kind, g) in [
        (KernelKind::Sa, &sub_milli),
        (KernelKind::Gibbs, &sub_milli),
        (KernelKind::Msa, &sub_milli),
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
    for kind in [KernelKind::Sa, KernelKind::Msa] {
        assert_stream_separates_scales(kind);
    }
}

fn assert_stream_separates_scales(kind: KernelKind) {
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
        let device = device(kind);
        let gov = UtilGovernor::start(&device.pci_bus_id, 90, false);
        run_stream(device, kind, job_rx, res_tx, CancelToken::default(), &gov);
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

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn sa_stream_mixes_unit_wide_refused_and_unit_jobs() {
    assert_stream_mixes_paths_and_refusal(KernelKind::Sa);
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn msa_stream_mixes_unit_weighted_refused_and_unit_jobs() {
    assert_stream_mixes_paths_and_refusal(KernelKind::Msa);
}

fn assert_stream_mixes_paths_and_refusal(kind: KernelKind) {
    use quip_solver_core::{CancelToken, SampleError as WireSampleError, StreamJob, StreamOutcome};
    use quip_solver_cuda::coefficients::{plan, CoefficientPath};
    use quip_solver_cuda::nvml_gov::UtilGovernor;
    use quip_solver_cuda::streaming::{run_stream, stream_width};
    use tokio::sync::mpsc::channel;

    let unit = IsingGraph::new(vec![0.0, 0.0], vec![1.0], vec![(0, 1)]);
    // Field 66 exceeds SA's int8 delta bound and MSA's unit coefficients.
    let mut fields = vec![0.0; 12];
    fields[0] = 64.0;
    let wide = IsingGraph::new(
        fields,
        vec![1.0; 12],
        (0..12).map(|i| (i, (i + 1) % 12)).collect(),
    );
    let refused = IsingGraph::new(vec![1e-19, 0.0], vec![1.0], vec![(0, 1)]);
    let dev = device(kind);
    let node_budget = match kind {
        KernelKind::Sa => dev.sa_wide_max_nodes,
        KernelKind::Msa => dev.max_nodes,
        KernelKind::Gibbs => unreachable!("this test covers SA and MSA"),
    };
    assert_eq!(
        plan(kind, &unit, node_budget).unwrap().path,
        CoefficientPath::Int8
    );
    assert_eq!(
        plan(kind, &wide, node_budget).unwrap().path,
        CoefficientPath::Wide
    );

    let mut graphs = vec![unit.clone()];
    // Fill cold-start admission so the refusal reaches the running session's pump.
    graphs.extend((0..stream_width(dev, kind)).map(|_| wide.clone()));
    let refused_id = graphs.len();
    graphs.extend([refused, unit]);
    let (job_tx, job_rx) = channel::<StreamJob>(graphs.len());
    let (res_tx, mut res_rx) = channel(16);
    for (id, graph) in graphs.iter().enumerate() {
        job_tx
            .try_send(StreamJob {
                job_id: vec![u8::try_from(id).unwrap()],
                graph: graph.clone(),
                params: SampleParams {
                    num_sweeps: 8192,
                    ..params()
                },
                watermark: None,
            })
            .unwrap();
    }
    drop(job_tx);
    let driver = std::thread::spawn(move || {
        let gov = UtilGovernor::start(&dev.pci_bus_id, 100, false);
        run_stream(dev, kind, job_rx, res_tx, CancelToken::default(), &gov);
    });
    let mut seen = vec![false; graphs.len()];
    while let Some(result) = res_rx.blocking_recv() {
        let id = usize::from(result.job_id[0]);
        assert!(!seen[id], "duplicate result for job {id}");
        seen[id] = true;
        let StreamOutcome::Completed(reads) = result.outcome else {
            panic!("job {id} was cancelled");
        };
        if id == refused_id {
            assert_eq!(reads.unwrap_err(), WireSampleError::Capacity);
        } else {
            let reads = reads.expect("accepted jobs in flight must complete");
            assert_eq!(reads.len(), params().num_reads);
            assert_ground_state(kind, &graphs[id], &reads);
        }
    }
    driver.join().expect("stream driver exits after draining");
    assert_eq!(seen, vec![true; graphs.len()], "one outcome for every job");
}

/// `K_n` with J in {-1, +1} from a fixed pattern and h = 0.
fn complete_pm1(n: usize) -> IsingGraph {
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            edges.push((u, v));
        }
    }
    let j = (0..edges.len())
        .map(|k| {
            if (k * 2_654_435_761) >> 7 & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    IsingGraph::new(vec![0.0; n], j, edges)
}

/// Best energy from single-spin descent out of 32 deterministic starts, in milli.
fn descent_reference(g: &IsingGraph) -> i64 {
    let n = g.h.len();
    (0..32_u64)
        .map(|s| {
            let mut spins: Vec<i8> = (0..n)
                .map(|i| {
                    let mut hash = (s << 32) | u64::try_from(i).expect("node index fits u64");
                    hash = hash.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    if (hash ^ (hash >> 31)) & 1 == 0 {
                        1
                    } else {
                        -1
                    }
                })
                .collect();
            let mut energy = energy_milli(&spins, &g.h, &g.j, &g.edges);
            loop {
                let mut improved = false;
                for i in 0..n {
                    spins[i] = -spins[i];
                    let flipped = energy_milli(&spins, &g.h, &g.j, &g.edges);
                    if flipped < energy {
                        energy = flipped;
                        improved = true;
                    } else {
                        spins[i] = -spins[i];
                    }
                }
                if !improved {
                    return energy;
                }
            }
        })
        .min()
        .unwrap()
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn complete_graphs_solve_on_both_paths() {
    // 16 and 64 run int8, 65 and 200 run wide (see coefficients::tests).
    for n in [16, 64, 65, 200] {
        let g = complete_pm1(n);
        let reads =
            sample_ising(device(KernelKind::Sa), &g, &params(), KernelKind::Sa).expect("solves");
        for r in &reads {
            assert_eq!(r.energy_milli, energy_milli(&r.spins, &g.h, &g.j, &g.edges));
        }
        let best = reads.iter().map(|r| r.energy_milli).min().unwrap();
        assert!(
            best <= descent_reference(&g),
            "K_{n}: {best} worse than descent"
        );
        if n <= 16 {
            assert_eq!(best, brute_force_min(&g));
        }
    }
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn large_and_wide_range_coefficients_solve_or_refuse() {
    // Wide range on the milli step: 0.001 next to 100000 units.
    let g = IsingGraph::new(
        vec![0.001, -100_000.0, 0.0, 3.5],
        vec![1.25, -0.5, 7.0, 0.001],
        vec![(0, 1), (1, 2), (2, 3), (3, 0)],
    );
    assert_finds_ground_state(KernelKind::Sa, &g);
    // Above the wide bound at scale 1: refused.
    let over = IsingGraph::new(vec![536_870.912, 0.0], vec![0.001], vec![(0, 1)]);
    let err = sample_ising(device(KernelKind::Sa), &over, &params(), KernelKind::Sa).unwrap_err();
    assert!(matches!(err, SampleError::Unsupported(_)));
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with make test-gpu"]
fn dense_quarter_unit_qubo_on_the_wide_path() {
    // K_12, quarter units up to ±2: field above 63 scale units, so wide.
    let n = 12;
    let vals = [-2.0, -1.25, -0.5, 0.25, 1.0, 1.75];
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            edges.push((u, v));
        }
    }
    let j = (0..edges.len()).map(|k| vals[(k * 7 + 2) % 6]).collect();
    let h = (0..n).map(|i| vals[(i * 5 + 1) % 6]).collect();
    let g = IsingGraph::new(h, j, edges);
    assert_finds_ground_state(KernelKind::Sa, &g);
    for lane in [MsaLane::I32, MsaLane::F32] {
        let (reads, _) = bench_one(
            device(KernelKind::Msa),
            &g,
            &params(),
            KernelKind::Msa,
            Some(lane),
        )
        .expect("weighted dense QUBO");
        assert_ground_state(KernelKind::Msa, &g, &reads);
    }
}

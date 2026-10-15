//! QUI-1598: many concurrent one-job `run_stream` sessions on one device.
//! `#[ignore]`: run under taskd with the variables below set.
//!
//! `QUIP_SESSIONS` (default 48), `QUIP_READS` (64), `QUIP_NODES` (2108),
//! `QUIP_SWEEPS` (8192), `QUIP_DEADLINE_SECS` (900).

use quip_solver_core::{
    CancelToken, IsingGraph, SampleParams, StreamJob, StreamOutcome, StreamResult,
};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::nvml_gov::UtilGovernor;
use quip_solver_cuda::streaming::run_stream;
use quip_solver_cuda::KernelKind;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Degree-4 ±1 lattice ring: within MSA's degree-20 limit, unit couplings.
fn model(n: usize) -> IsingGraph {
    let mut edges = Vec::new();
    for i in 0..n {
        edges.push((i, (i + 1) % n));
        edges.push((i, (i + 7) % n));
    }
    let j = (0..edges.len())
        .map(|k| if k % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    IsingGraph::new(vec![0.0; n], j, edges)
}

#[test]
#[ignore = "QUI-1598 reproduction; run under taskd on a GPU host"]
fn concurrent_one_job_sessions_all_finish() {
    let sessions = env("QUIP_SESSIONS", 48);
    let reads = env("QUIP_READS", 64);
    let n = env("QUIP_NODES", 2108);
    let sweeps = env("QUIP_SWEEPS", 8192);
    let deadline = Duration::from_secs(
        u64::try_from(env("QUIP_DEADLINE_SECS", 900)).expect("deadline fits u64"),
    );

    let device =
        Arc::new(CudaDevice::open_with_nodes(0, KernelKind::Msa, n.max(5000)).expect("device 0"));
    let gov = Arc::new(UtilGovernor::start(&device.pci_bus_id, 100, false));
    let (done_tx, done_rx) = mpsc::channel::<(usize, Duration, bool)>();
    let start = Instant::now();

    for s in 0..sessions {
        let (device, gov, done_tx) = (Arc::clone(&device), Arc::clone(&gov), done_tx.clone());
        let graph = model(n);
        std::thread::spawn(move || {
            let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
            let (res_tx, mut res_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);
            job_tx
                .blocking_send(StreamJob {
                    job_id: s.to_le_bytes().to_vec(),
                    graph,
                    params: SampleParams {
                        num_reads: reads,
                        num_sweeps: sweeps,
                        ..SampleParams::default()
                    },
                    watermark: None,
                })
                .unwrap();
            drop(job_tx);
            run_stream(
                &device,
                KernelKind::Msa,
                job_rx,
                res_tx,
                CancelToken::default(),
                &gov,
            );
            let ok = res_rx
                .blocking_recv()
                .is_some_and(|r| matches!(r.outcome, StreamOutcome::Completed(Ok(_))));
            let _sent = done_tx.send((s, start.elapsed(), ok));
        });
    }
    drop(done_tx);

    let mut finished = Vec::new();
    while finished.len() < sessions {
        match done_rx.recv_timeout(deadline.saturating_sub(start.elapsed())) {
            Ok(entry) => finished.push(entry),
            Err(_) => break,
        }
    }
    eprintln!(
        "QUI-1598: {}/{sessions} sessions finished in {:?}",
        finished.len(),
        start.elapsed()
    );
    for (s, t, ok) in &finished {
        eprintln!("  session {s}: {t:?} ok={ok}");
    }
    assert_eq!(finished.len(), sessions, "sessions hung");
    assert!(
        finished.iter().all(|(_, _, ok)| *ok),
        "a session returned an error"
    );
}

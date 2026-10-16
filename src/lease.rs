//! Phased MSA lease screening and finite full-solve batches.

use crate::screen::{Screen, ScreenDomain, ScreenError, ScreenParams};
use crate::screen_policy::{Decision, SelectorKey};
use crate::{streaming, CudaSampler, IsingGraph, KernelKind, MsaSampler, SampleParams};
use quip_solver_core::{
    CancelToken, Lease, LeaseSink, SampleError, StreamJob, StreamOutcome, StreamResult,
    TopologyView,
};
use std::collections::BTreeMap;
use std::sync::{Arc, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{channel, Receiver};

fn fault(error: impl std::fmt::Display) -> SampleError {
    SampleError::DeviceFault(format!("CUDA lease: {error}"))
}

fn chunk_size(width: usize, screen: Option<(u32, usize)>, remaining: u64) -> usize {
    let base = width.max(1).saturating_mul(2);
    let count = screen.map_or(base, |(keep, budget)| {
        base.saturating_mul(usize::try_from(keep).unwrap_or(usize::MAX))
            .min(budget.max(1))
    });
    count.min(usize::try_from(remaining).unwrap_or(usize::MAX))
}

struct ProbeBudget {
    salts: usize,
}

impl ProbeBudget {
    fn new(params: ScreenParams) -> Self {
        // Start conservatively at one group per 21 ms for 512 sweeps on the
        // A4000. Later chunks use observed wall time, including device sharing.
        let group_ms = (21 * params.sweeps).div_ceil(512).max(1);
        Self {
            salts: (1000 / group_ms).max(1) * (32 >> params.reads_log2),
        }
    }

    fn observe(&mut self, salts: usize, elapsed: Duration) {
        self.salts = usize::try_from(
            (salts as u128).saturating_mul(1_000_000_000) / elapsed.as_nanos().max(1),
        )
        .unwrap_or(usize::MAX)
        .max(1);
    }
}

fn salt_index(job_id: &[u8]) -> Result<u64, SampleError> {
    let bytes = job_id
        .try_into()
        .map_err(|_| fault("invalid salt job id"))?;
    Ok(u64::from_le_bytes(bytes))
}

/// Only final rejects can be accounted before a full solve.
fn admit(pending: &mut BTreeMap<u64, Decision>, index: u64, decision: Decision) -> bool {
    match decision {
        Decision::Reject => true,
        Decision::Keep | Decision::Audit => {
            pending.insert(index, decision);
            false
        }
    }
}

fn take_decision(
    pending: &mut BTreeMap<u64, Decision>,
    job_id: &[u8],
) -> Result<(u64, Decision), SampleError> {
    let index = salt_index(job_id)?;
    let decision = pending
        .remove(&index)
        .ok_or_else(|| fault("unknown or repeated salt result"))?;
    Ok((index, decision))
}

struct Stats {
    mode: &'static str,
    assigned: u64,
    turn: u64,
    started: Instant,
    screened: u64,
    kept: u64,
    audited: u64,
    pushed: u64,
    probe: Duration,
    full: Duration,
}

impl Stats {
    fn new(assigned: u64, turn: u64) -> Self {
        Self {
            mode: "Waiting",
            assigned,
            turn,
            started: Instant::now(),
            screened: 0,
            kept: 0,
            audited: 0,
            pushed: 0,
            probe: Duration::ZERO,
            full: Duration::ZERO,
        }
    }
}

impl Drop for Stats {
    fn drop(&mut self) {
        let salts = self.screened + self.pushed;
        #[allow(clippy::cast_precision_loss)] // Reporting a count per wall-clock second.
        let salts_per_s = salts as f64 / self.started.elapsed().as_secs_f64().max(f64::EPSILON);
        tracing::info!(
            mode = self.mode,
            turn = self.turn,
            salts,
            assigned = self.assigned,
            screened = self.screened,
            kept = self.kept,
            audited = self.audited,
            pushed = self.pushed,
            probe_ms = self.probe.as_millis(),
            full_ms = self.full.as_millis(),
            salts_per_s,
            "CUDA lease finished"
        );
    }
}

pub(crate) fn sample(
    sampler: &MsaSampler,
    lease: &Lease,
    topology: &TopologyView,
    params: &SampleParams,
    sink: &LeaseSink,
) -> Result<(), SampleError> {
    let turn = sampler.turns.join();
    let mut stats = Stats::new(lease.salt_count(), turn.id());
    if !turn.wait(|| sink.is_stopped()) || sink.is_stopped() {
        return Ok(());
    }
    let config = sampler
        .inner
        .screen
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .config();
    // Core 0.0.3 exposes no wire hash to local samplers. The key includes the
    // exact ordered edges and node count, so distinct graphs cannot share state.
    let key = SelectorKey::new(&[], topology, config.params());
    let opened = if config.enabled {
        match ScreenDomain::check(&sampler.inner.device, topology, config.params()) {
            Ok(shape) => Screen::open(&sampler.inner.device, shape).map(Some),
            Err(reason) => Err(ScreenError::NotScreenable(reason)),
        }
    } else {
        Ok(None)
    };
    let (screen, reason) = match opened {
        Ok(Some(screen)) => (Some(screen), "eligible".to_owned()),
        Ok(None) => (None, "screen disabled".to_owned()),
        Err(ScreenError::NotScreenable(reason)) => (None, reason.to_string()),
        Err(error) => return Err(fault(error)),
    };
    stats.mode = if screen.is_some() {
        "Screened"
    } else {
        "Direct"
    };
    if sampler
        .logged_modes
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key.clone())
    {
        tracing::info!(mode = stats.mode, %reason, nodes = topology.num_nodes, edges = topology.edges.len(), "CUDA lease topology mode");
    }
    work(
        sampler,
        lease,
        topology,
        params,
        sink,
        screen.as_ref(),
        &key,
        &mut stats,
    )
}

#[allow(clippy::too_many_arguments)] // Lease context plus its optional probe and counters.
fn work(
    sampler: &MsaSampler,
    lease: &Lease,
    topology: &TopologyView,
    params: &SampleParams,
    sink: &LeaseSink,
    screen: Option<&Screen<'_>>,
    key: &SelectorKey,
    stats: &mut Stats,
) -> Result<(), SampleError> {
    let width = streaming::stream_width(&sampler.inner.device, KernelKind::Msa);
    let mut budget = ProbeBudget::new(screen.map_or_else(ScreenParams::default, Screen::params));
    let mut start = 0;
    while start < lease.salt_count() && !sink.is_stopped() {
        let config = sampler
            .inner
            .screen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .config();
        let count = chunk_size(
            width,
            screen.map(|_| (config.keep_denominator, budget.salts)),
            lease.salt_count() - start,
        );
        let end = start + u64::try_from(count).map_err(fault)?;
        let mut pending = BTreeMap::new();
        if let Some(screen) = screen {
            let nonces: Vec<_> = (start..end).map(|index| lease.nonce(index)).collect();
            let probe_start = Instant::now();
            let run = screen.run(&nonces, &|| sink.is_stopped());
            let elapsed = probe_start.elapsed();
            stats.probe += elapsed;
            let run = run.map_err(fault)?;
            budget.observe(run.best.len(), elapsed);
            let target = sink.target_energy_milli();
            for (offset, energy) in run.best.into_iter().enumerate() {
                if sink.is_stopped() {
                    return Ok(());
                }
                let index = start + u64::try_from(offset).map_err(fault)?;
                let decision = sampler
                    .inner
                    .screen
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .decide(key, energy, target);
                if admit(&mut pending, index, decision) {
                    if sink.screen(index, energy).is_err() {
                        return Ok(());
                    }
                    stats.screened += 1;
                } else if decision == Decision::Audit {
                    stats.audited += 1;
                } else {
                    stats.kept += 1;
                }
            }
            if run.stopped || sink.is_stopped() {
                return Ok(());
            }
        } else {
            for index in start..end {
                pending.insert(index, Decision::Keep);
            }
            stats.kept += u64::try_from(count).map_err(fault)?;
        }
        if !pending.is_empty() && !sink.is_stopped() {
            let full_start = Instant::now();
            let solved = full_solve(
                &sampler.inner,
                lease,
                topology,
                params,
                sink,
                screen.map(|_| key),
                pending,
                &mut stats.pushed,
            );
            stats.full += full_start.elapsed();
            if solved? {
                return Ok(());
            }
        }
        tracing::debug!(start, end, "CUDA lease chunk finished");
        start = end;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Immutable lease inputs and the selected salt decisions.
fn full_solve(
    sampler: &CudaSampler,
    lease: &Lease,
    topology: &TopologyView,
    params: &SampleParams,
    sink: &LeaseSink,
    key: Option<&SelectorKey>,
    pending: BTreeMap<u64, Decision>,
    pushed: &mut u64,
) -> Result<bool, SampleError> {
    let (job_tx, job_rx) = channel(pending.len());
    let batch_len = pending.len();
    let edges: Arc<[(usize, usize)]> = topology.edges.clone().into();
    for &index in pending.keys() {
        if sink.is_stopped() {
            return Ok(true);
        }
        let (h, j) = topology.draw(lease.nonce(index)).map_err(fault)?;
        let graph = IsingGraph {
            h: h.into_iter().map(|v| f64::from(v) / 1000.0).collect(),
            j: j.into_iter().map(|v| f64::from(v) / 1000.0).collect(),
            edges: Arc::clone(&edges),
        };
        job_tx
            .try_send(StreamJob {
                job_id: index.to_le_bytes().to_vec(),
                graph,
                params: params.clone(),
                watermark: Some(1),
            })
            .map_err(fault)?;
    }
    // Closing intake lets even a sparse finite batch skip the cold-start timeout.
    drop(job_tx);
    run_full_batch(
        sampler,
        job_rx,
        batch_len,
        || sink.is_stopped(),
        |out, cancel| report(sampler, sink, key, pending, out, cancel, pushed),
    )
}

fn run_full_batch(
    sampler: &CudaSampler,
    jobs: Receiver<StreamJob>,
    batch_len: usize,
    stopped: impl Fn() -> bool + Sync,
    report: impl FnOnce(Receiver<StreamResult>, &CancelToken) -> Result<bool, SampleError> + Send,
) -> Result<bool, SampleError> {
    // One result per job fits even while the reporter is blocked in core.
    let (out_tx, out_rx) = channel(batch_len);
    let cancel = CancelToken::default();
    std::thread::scope(|scope| {
        let reporter = scope.spawn(|| report(out_rx, &cancel));
        let (session_done, retired) = std::sync::mpsc::channel::<()>();
        let cancel_ref = &cancel;
        let stop_ref = &stopped;
        let watcher = scope.spawn(move || loop {
            if stop_ref() {
                cancel_ref.cancel_through(1);
                break;
            }
            match retired.recv_timeout(Duration::from_millis(10)) {
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        });
        streaming::run_stream(
            &sampler.device,
            KernelKind::Msa,
            jobs,
            out_tx,
            cancel.clone(),
            &sampler.gov,
        );
        drop(session_done);
        watcher
            .join()
            .map_err(|_| fault("lease stop watcher panicked"))?;
        reporter
            .join()
            .map_err(|_| fault("lease reporter panicked"))?
    })
}

fn report(
    sampler: &CudaSampler,
    sink: &LeaseSink,
    key: Option<&SelectorKey>,
    mut pending: BTreeMap<u64, Decision>,
    mut out: Receiver<StreamResult>,
    cancel: &CancelToken,
    pushed: &mut u64,
) -> Result<bool, SampleError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .map_err(fault)?;
    let mut stopped = false;
    let mut failure = None;
    loop {
        stopped |= sink.is_stopped();
        if stopped || failure.is_some() {
            cancel.cancel_through(1);
        }
        // Only receive inside the runtime. LeaseSink::push uses blocking_send
        // and must run outside an asynchronous runtime context.
        let received = runtime
            .block_on(async { tokio::time::timeout(Duration::from_millis(10), out.recv()).await });
        let result = match received {
            Ok(Some(result)) => result,
            Ok(None) => break,
            Err(_) => continue,
        };
        stopped |= sink.is_stopped();
        if stopped || failure.is_some() {
            // Keep draining after a stop, but never publish remaining results.
            continue;
        }
        let mapped = take_decision(&mut pending, &result.job_id);
        let completed = match (mapped, result.outcome) {
            (Ok((index, decision)), StreamOutcome::Completed(Ok(reads))) => {
                Some((index, decision, reads))
            }
            (Ok(_), StreamOutcome::Cancelled) => {
                stopped = true;
                None
            }
            (Err(error), _) | (_, StreamOutcome::Completed(Err(error))) => {
                failure.get_or_insert_with(|| fault(error));
                cancel.cancel_through(1);
                None
            }
        };
        stopped |= sink.is_stopped();
        if failure.is_some() || stopped {
            continue;
        }
        if let Some((index, decision, reads)) = completed {
            let energy = reads.iter().map(|r| r.energy_milli).min();
            if energy.is_none() {
                failure = Some(fault("full solve returned no reads"));
                cancel.cancel_through(1);
                continue;
            }
            // Core's blocking push cannot be interrupted. The independent
            // watcher retires the device session on stop, but the lease thread
            // can still wait for an in-flight push to return.
            if sink.push(index, reads).is_err() {
                stopped = true;
                cancel.cancel_through(1);
                continue;
            }
            *pushed += 1;
            if let (Some(key), Some(energy)) = (key, energy) {
                let observation = sampler
                    .screen
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .record_full(key, decision, energy);
                if let Some(observation) = observation {
                    tracing::debug!(?observation, "CUDA lease audit");
                }
            }
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    if !pending.is_empty() && !stopped && !sink.is_stopped() {
        return Err(fault("full solve closed before every salt returned"));
    }
    Ok(stopped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::Sampler;

    #[test]
    #[ignore = "requires CUDA device 0"]
    fn blocked_push_does_not_prevent_device_session_retirement() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let device = crate::cuda_device::CudaDevice::open(0).unwrap();
        let gov = crate::nvml_gov::UtilGovernor::start(&device.pci_bus_id, 100, false);
        let sampler = CudaSampler::new(device, gov, KernelKind::Msa);
        let count = sampler.stream_width() * 3;
        let (jobs, incoming) = channel(count);
        let n = 1024;
        for index in 0..count {
            jobs.try_send(StreamJob {
                job_id: index.to_le_bytes().to_vec(),
                graph: IsingGraph::new(
                    vec![0.0; n],
                    vec![1.0; n],
                    (0..n).map(|i| (i, (i + 1) % n)).collect(),
                ),
                params: SampleParams {
                    num_reads: 64,
                    num_sweeps: if index == 0 { 32 } else { 20_000_000 },
                    ..SampleParams::default()
                },
                watermark: Some(1),
            })
            .unwrap();
        }
        drop(jobs);
        let stopped = AtomicBool::new(false);
        let (pushing, push_started) = std::sync::mpsc::channel();
        let (_release, blocked_push) = std::sync::mpsc::channel::<()>();
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                run_full_batch(
                    &sampler,
                    incoming,
                    count,
                    || stopped.load(Ordering::Acquire),
                    move |mut results, cancel| {
                        let first = results.blocking_recv().expect("first solve");
                        assert!(matches!(first.outcome, StreamOutcome::Completed(Ok(_))));
                        pushing.send(()).unwrap();
                        // Simulate a core push that cannot return for five seconds.
                        assert!(blocked_push.recv_timeout(Duration::from_secs(5)).is_err());
                        let retired = results.is_closed();
                        // Always unblock the old implementation after the assertion
                        // snapshot, so a regression fails without leaving GPU work.
                        cancel.cancel_through(1);
                        let mut received = 1;
                        let mut cancelled = 0;
                        while let Some(result) = results.blocking_recv() {
                            received += 1;
                            if matches!(result.outcome, StreamOutcome::Cancelled) {
                                cancelled += 1;
                            }
                        }
                        assert!(retired, "run_stream must return while push is blocked");
                        assert_eq!(
                            received, count,
                            "every local job can retire without a receiver"
                        );
                        assert!(cancelled > 0, "stop must abandon the long full solves");
                        Ok(true)
                    },
                )
            });
            let started = push_started.recv_timeout(Duration::from_secs(30));
            stopped.store(true, Ordering::Release);
            let outcome = worker.join();
            started.expect("first solve reaches the blocked push");
            assert!(outcome.unwrap().unwrap());
        });
    }

    #[test]
    fn chunks_cover_the_remaining_range_and_target_two_widths() {
        assert_eq!(chunk_size(48, None, 1000), 96);
        assert_eq!(chunk_size(48, None, 3), 3);
        assert_eq!(chunk_size(48, Some((64, 10_000)), 100_000), 6144);
        assert_eq!(chunk_size(48, Some((64, 376)), 100_000), 376);
        assert_eq!(chunk_size(48, Some((64, 376)), 0), 0);
        assert_eq!(chunk_size(0, None, 8), 2);
    }

    #[test]
    fn probe_budget_scales_with_shape_and_measured_time() {
        let mut budget = ProbeBudget::new(ScreenParams::default());
        assert_eq!(budget.salts, 376);
        budget.observe(100, Duration::from_millis(500));
        assert_eq!(budget.salts, 200);
        budget.observe(1, Duration::from_secs(2));
        assert_eq!(budget.salts, 1);
        let deep = ProbeBudget::new(ScreenParams {
            sweeps: 2048,
            ..ScreenParams::default()
        });
        assert_eq!(deep.salts, 88);
    }

    #[test]
    fn job_ids_preserve_all_eight_little_endian_bytes() {
        for index in [0, 255, 256, 1 << 40, u64::MAX] {
            assert_eq!(salt_index(&index.to_le_bytes()).unwrap(), index);
        }
        assert!(salt_index(&[]).is_err());
        assert!(salt_index(&[0; 9]).is_err());
    }

    #[test]
    fn only_final_rejects_are_screened_and_full_outcomes_keep_their_decision() {
        let mut pending = BTreeMap::new();
        assert!(admit(&mut pending, 0, Decision::Reject));
        assert!(!admit(&mut pending, 1, Decision::Keep));
        assert!(!admit(&mut pending, 2, Decision::Audit));
        assert_eq!(pending.len(), 2);
        assert_eq!(
            take_decision(&mut pending, &2_u64.to_le_bytes()).unwrap(),
            (2, Decision::Audit)
        );
        assert!(take_decision(&mut pending, &2_u64.to_le_bytes()).is_err());
        assert!(take_decision(&mut pending, &0_u64.to_le_bytes()).is_err());
        assert_eq!(
            take_decision(&mut pending, &1_u64.to_le_bytes()).unwrap(),
            (1, Decision::Keep)
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn only_msa_opts_into_local_generation() {
        assert!(crate::MsaSampler::generates_locally());
        assert!(!CudaSampler::generates_locally());
        assert_eq!(
            crate::MsaSampler::declared_stream_width(),
            CudaSampler::declared_stream_width()
        );
        assert_eq!(
            crate::MsaSampler::accepts_warm_start(),
            CudaSampler::accepts_warm_start()
        );
    }
}

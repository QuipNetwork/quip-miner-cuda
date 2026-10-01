//! GPU-gated check of abort-on-cancel in the streaming driver.
//!
//! Ignored by default: run with
//! `cargo test --release --test cancel_gpu -- --ignored`.
//!
//! Both tests open device 0 and leave a persistent kernel resident. They are
//! `serial` so those kernels are not launched together.

use quip_miner_cuda::capacity::MSA_DEFAULT_NODES;
use quip_miner_cuda::cuda_device::CudaDevice;
use quip_miner_cuda::nvml_gov::UtilGovernor;
use quip_miner_cuda::streaming::run_stream;
use quip_miner_cuda::KernelKind;
use quip_solver_core::{
    CancelToken, IsingGraph, SampleParams, StreamJob, StreamOutcome, StreamResult,
};
use serial_test::serial;
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::channel;

/// Sweeps for the job that gets cancelled. Far longer than `ABORT_DEADLINE`
/// on any supported card, so a `Cancelled` inside the deadline can only come
/// from the abort, not from the model finishing.
const LONG_SWEEPS: usize = 20_000_000;
/// Time from the launch to the cancel, so the long job is active in a slot
/// rather than still queued (a queued job is refunded at dequeue instead).
const SETTLE: Duration = Duration::from_secs(2);
/// Upper bound on cancel-to-refund latency.
const ABORT_DEADLINE: Duration = Duration::from_secs(30);

/// 1024-node ring with alternating couplings: degree 2, so it fits SA and msa.
fn ring() -> IsingGraph {
    let n = 1024;
    let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    let j = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    IsingGraph::new(vec![0.0; n], j, edges)
}

fn job(id: u8, reads: usize, sweeps: usize, watermark: u64) -> StreamJob {
    StreamJob {
        job_id: vec![id],
        graph: ring(),
        params: SampleParams {
            num_reads: reads,
            num_sweeps: sweeps,
            sweeps_per_beta: 1,
            beta_range: None,
            seed: u64::from(id),
        },
        watermark: Some(watermark),
    }
}

/// Start a long job, cancel its round once it is running, then send a job
/// from the next round with a different sweep count so it seeds a new
/// session. The long job must come back `Cancelled` inside the deadline, and
/// the next job must complete with samples.
fn cancel_then_complete(device: &CudaDevice, kernel: KernelKind, reads: usize) {
    let gov = UtilGovernor::start(&device.pci_bus_id, 100, false);
    let cancel = CancelToken::default();
    let (job_tx, job_rx) = channel::<StreamJob>(4);
    let (res_tx, mut res_rx) = channel::<StreamResult>(4);

    thread::scope(|s| {
        let driver_cancel = cancel.clone();
        let gov_ref = &gov;
        s.spawn(move || run_stream(device, kernel, job_rx, res_tx, driver_cancel, gov_ref));

        job_tx
            .blocking_send(job(1, reads, LONG_SWEEPS, 1))
            .expect("driver accepts the long job");
        thread::sleep(SETTLE);

        let cancelled_at = Instant::now();
        cancel.cancel_through(1);
        job_tx
            .blocking_send(job(2, reads, 64, 2))
            .expect("driver accepts the next round");
        drop(job_tx);

        let mut long = None;
        let mut next = None;
        while let Some(r) = res_rx.blocking_recv() {
            match r.job_id[0] {
                1 => long = Some((r.outcome, cancelled_at.elapsed())),
                2 => next = Some(r.outcome),
                other => panic!("unexpected job id {other}"),
            }
        }

        let (outcome, latency) = long.expect("the long job must produce a result");
        assert!(
            matches!(outcome, StreamOutcome::Cancelled),
            "{kernel:?}: the long job must be refunded as Cancelled, not published"
        );
        assert!(
            latency < ABORT_DEADLINE,
            "{kernel:?}: abort took {latency:?}, over {ABORT_DEADLINE:?}"
        );
        let Some(StreamOutcome::Completed(Ok(samples))) = next else {
            panic!("{kernel:?}: the next-round job must complete with samples");
        };
        assert!(
            !samples.is_empty(),
            "{kernel:?}: no samples from the next job"
        );
    });
}

/// SA with fewer than 256 reads: a partial block, the case the bead names.
#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn sa_cancel_aborts_the_active_model_and_the_next_session_completes() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }
    let device = CudaDevice::open(0).expect("open SA");
    cancel_then_complete(&device, KernelKind::Sa, 64);
}

/// msa at 128 reads, two replica words. Skips on a one-word device.
#[test]
#[serial]
#[ignore = "requires a CUDA GPU"]
fn msa_cancel_aborts_the_active_model_and_the_next_session_completes() {
    if CudaDevice::device_count().unwrap_or(0) == 0 {
        eprintln!("no CUDA device visible; skipping");
        return;
    }
    let device =
        CudaDevice::open_with_nodes(0, KernelKind::Msa, MSA_DEFAULT_NODES).expect("open msa");
    if device.msa_replica_words < 2 {
        eprintln!("device holds one msa replica word; 128 reads not servable, skipping");
        return;
    }
    cancel_then_complete(&device, KernelKind::Msa, 128);
}

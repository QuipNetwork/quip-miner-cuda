//! Salt leases through solver-core's real session and the MSA binary.
//! Run on device 0 with `--release --ignored --test-threads=1`.

mod common;
#[path = "common/lease.rs"]
mod support;

use common::{ensure_built, profile_bin};
use quip_solver_core::quip_proto::v1::{
    coord_msg, ising_problem, Cancel, CoefficientEncoding, EdgeList, IsingProblem,
    IsingProblemGenerator, Job, JobKind, RejectReason,
};
use quip_solver_core::quip_protocol::lease::{verify_lease_result, TopologyView};
use quip_solver_core::quip_protocol::wire::encode_i32_le;
use std::collections::HashSet;
use std::time::Duration;
use support::{lease, permissive_target, ring, target, Session};

const RING_A: [u8; 32] = [0x51; 32];
const RING_B: [u8; 32] = [0x52; 32];
const LONG: Duration = Duration::from_secs(90);

async fn start(config: &str) -> Session {
    ensure_built(&["quip-cuda-msa"]);
    Session::start(&profile_bin("quip-cuda-msa"), 4, config).await
}

fn generator(job: &Job) -> IsingProblemGenerator {
    job.generator.clone().expect("lease generator")
}

fn verify(s: &Session, id: &[u8], spec: &IsingProblemGenerator, view: &TopologyView, count: u64) {
    assert_eq!(s.log.done(id).expect("LeaseDone").salts_done, count);
    assert_eq!(s.log.done_count(id), 1, "one summary");
    let mut salts = HashSet::new();
    for result in s.log.results_for(id) {
        // The verifier derives the nonce, redraws the canonical milli model,
        // decodes every spin, and checks each claimed energy independently.
        let verified = verify_lease_result(spec, view, &permissive_target(), result)
            .expect("independent nonce, problem and energy verification");
        assert!(salts.insert(verified.salt), "one result per salt");
        let meta = result.meta.as_ref().expect("sample metadata");
        assert_eq!(result.solutions.len(), usize::try_from(meta.reads).unwrap());
        assert!(
            meta.reads == 64 || meta.reads == 128,
            "MSA resolved read count"
        );
    }
}

async fn finish(s: &mut Session, id: &[u8]) {
    s.until("lease summary and credit", LONG, |log| {
        log.done(id).is_some() && log.refunds() >= u32::try_from(log.lease_done.len()).unwrap()
    })
    .await;
    assert!(
        s.log.rejects.is_empty(),
        "{:?}\n{}",
        s.log.rejects,
        s.stderr()
    );
    assert!(s.log.fatal.is_none(), "{:?}\n{}", s.log.fatal, s.stderr());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn every_salt_is_counted_once_and_every_full_result_verifies() {
    let mut s = start("[screen]\naudit_denominator = 2").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 64)))
        .await;
    let measurement = lease(b"measure", 3, RING_A, 100, 32);
    let spec = generator(&measurement);
    s.send(coord_msg::Msg::Job(measurement)).await;
    finish(&mut s, b"measure").await;
    verify(&s, b"measure", &spec, &view, 32);
    assert_eq!(s.log.results_for(b"measure").len(), 32);
    let mut bests: Vec<_> = s
        .log
        .results_for(b"measure")
        .iter()
        .map(|r| r.solutions.iter().map(|x| x.energy_milli).min().unwrap())
        .collect();
    bests.sort_unstable();
    s.send(coord_msg::Msg::SetTarget(target(
        bests[bests.len() / 2],
        64,
    )))
    .await;
    let job = lease(b"median", 5, RING_A, 1000, 64);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    finish(&mut s, b"median").await;
    verify(&s, b"median", &spec, &view, 64);
    assert_eq!(s.jobs_done().await, 96, "no repeated accounting");
    assert_eq!(s.shutdown(2000).await, 0);
    assert!(s.stderr().contains("Screened"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn an_easy_target_and_a_lease_only_session_report_every_salt() {
    let mut s = start("").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    let job = lease(b"all", 3, RING_A, 0, 24);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    finish(&mut s, b"all").await;
    verify(&s, b"all", &spec, &view, 24);
    assert_eq!(s.log.results_for(b"all").len(), 24);
    assert_eq!(s.shutdown(2000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn a_target_change_during_a_lease_applies_to_later_chunks() {
    let mut s = start("[screen]\naudit_denominator = 0\nprobe_sweeps = 2048").await;
    let (topology, view) = ring(1024, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MIN, 32)))
        .await;
    let job = lease(b"live-target", 3, RING_A, 0, 1024);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    // The initial probe budget is 88 salts at 2048 sweeps. Those warm-up
    // observations are all rejects. Wait for the actual chunk boundary.
    s.until_stderr("CUDA lease chunk finished", LONG).await;
    assert!(s.log.done(b"live-target").is_none());
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    finish(&mut s, b"live-target").await;
    verify(&s, b"live-target", &spec, &view, 1024);
    let salts: HashSet<_> = s
        .log
        .results_for(b"live-target")
        .iter()
        .map(|result| u64::from_le_bytes(result.salt[..8].try_into().unwrap()))
        .collect();
    assert!(
        (0..88).all(|salt| !salts.contains(&salt)),
        "unreachable warm-up chunk returns no reads"
    );
    assert!(
        (896..1024).all(|salt| salts.contains(&salt)),
        "easy target keeps every salt in later chunks"
    );
    assert_eq!(s.shutdown(2000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn unreachable_target_counts_screened_and_audited_salts() {
    let mut s = start("[screen]\naudit_denominator = 2").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(-128_001, 64)))
        .await;
    let job = lease(b"none", 3, RING_A, 1 << 40, 96);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    finish(&mut s, b"none").await;
    verify(&s, b"none", &spec, &view, 96);
    let count = s.log.results_for(b"none").len();
    assert!(
        count > 0 && count < 96,
        "audits produce reads, rejects do not"
    );
    assert!(s.log.done(b"none").unwrap().best_energy_milli > -128_001);
    assert_eq!(s.jobs_done().await, 96);
    assert_eq!(s.shutdown(2000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn cancel_cuts_a_full_solve_short_with_one_summary_and_one_refund() {
    let mut s = start("").await;
    let (topology, _) = ring(1024, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 16_384)))
        .await;
    s.send(coord_msg::Msg::Job(lease(
        b"cancel", 7, RING_A, 0, 1_000_000,
    )))
    .await;
    s.until("first full solve", LONG, |log| {
        !log.results_for(b"cancel").is_empty()
    })
    .await;
    s.send(coord_msg::Msg::Cancel(Cancel { max_generation: 7 }))
        .await;
    finish(&mut s, b"cancel").await;
    assert!(s.log.done(b"cancel").unwrap().salts_done < 1_000_000);
    assert_eq!(s.log.done_count(b"cancel"), 1);
    assert_eq!(s.log.refunds(), 1);
    // The next turn must work after the cancelled stream and its buffers retire.
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"after", 8, RING_A, 0, 3)))
        .await;
    finish(&mut s, b"after").await;
    assert_eq!(s.log.results_for(b"after").len(), 3);
    assert_eq!(s.shutdown(2000).await, 0);
    assert_eq!(s.log.done_count(b"cancel"), 1);
    assert_eq!(s.log.refunds(), 2);
    assert_eq!(s.stderr().matches("CUDA lease finished").count(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn shutdown_during_a_lease_sends_one_summary_and_refund() {
    let mut s = start("").await;
    let (topology, _) = ring(1024, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 16_384)))
        .await;
    s.send(coord_msg::Msg::Job(lease(
        b"shutdown",
        9,
        RING_A,
        0,
        1_000_000,
    )))
    .await;
    s.until("first result", LONG, |log| {
        !log.results_for(b"shutdown").is_empty()
    })
    .await;
    assert_eq!(s.shutdown(3000).await, 0);
    assert_eq!(s.log.done_count(b"shutdown"), 1);
    assert_eq!(s.log.refunds(), 1);
    let count = s.log.done(b"shutdown").unwrap().salts_done;
    assert!(count > 0 && count < 1_000_000);
    let summary = s
        .log
        .order
        .iter()
        .position(|e| e == "done:shutdown")
        .unwrap();
    assert!(!s.log.order[summary..]
        .iter()
        .any(|e| e == "result:shutdown"));
    assert_eq!(s.stderr().matches("CUDA lease finished").count(), 1);
    assert!(!s.stderr().contains("exceeded the shutdown grace"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn a_plain_job_during_a_lease_keeps_its_read_count() {
    let mut s = start("").await;
    let (topology, _) = ring(1024, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 8192)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"long", 3, RING_A, 0, 1_000_000)))
        .await;
    s.until("lease active", LONG, |log| {
        !log.results_for(b"long").is_empty()
    })
    .await;
    s.send(coord_msg::Msg::Job(Job {
        job_id: b"plain".to_vec(),
        kind: JobKind::IsingSample as i32,
        generation: 3,
        ising: Some(IsingProblem {
            encoding: CoefficientEncoding::I32 as i32,
            scale: 1000,
            h: encode_i32_le(&[1000, -1000, 0]),
            j: encode_i32_le(&[1000, -1000]),
            num_reads: 16,
            num_sweeps: 64,
            graph: Some(ising_problem::Graph::Edges(EdgeList {
                u: vec![0, 1],
                v: vec![1, 2],
            })),
            ..Default::default()
        }),
        ..Default::default()
    }))
    .await;
    s.until("plain result", LONG, |log| {
        !log.results_for(b"plain").is_empty()
    })
    .await;
    let results = s.log.results_for(b"plain");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].solutions.len(), 16);
    assert_eq!(results[0].meta.as_ref().unwrap().sweeps, 64);
    assert_eq!(s.shutdown(2000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn missing_target_and_topology_mismatch_reject_with_refunds() {
    let mut s = start("").await;
    let (topology, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::Job(lease(b"no-target", 3, RING_A, 0, 10)))
        .await;
    s.until("missing target", LONG, |log| !log.rejects.is_empty())
        .await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"wrong-hash", 3, RING_B, 0, 10)))
        .await;
    s.until("rejects and credits", LONG, |log| {
        log.rejects.len() == 2 && log.refunds() == 2
    })
    .await;
    for (id, reason) in [
        (b"no-target".as_slice(), RejectReason::TargetMissing),
        (b"wrong-hash".as_slice(), RejectReason::TopologyMismatch),
    ] {
        assert_eq!(
            s.log
                .rejects
                .iter()
                .find(|r| r.job_id == id)
                .unwrap()
                .reason,
            reason as i32
        );
    }
    assert_eq!(s.shutdown(2000).await, 0);
    assert!(s.log.lease_done.is_empty());
    assert_eq!(s.log.refunds(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn topology_replacement_uses_the_new_graph() {
    let mut s = start("").await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    for (id, hash, nodes) in [(b"a".as_slice(), RING_A, 64), (b"b".as_slice(), RING_B, 96)] {
        let (topology, view) = ring(nodes, hash);
        s.send(coord_msg::Msg::Topology(topology)).await;
        let job = lease(id, 3, hash, 0, 8);
        let spec = generator(&job);
        s.send(coord_msg::Msg::Job(job)).await;
        finish(&mut s, id).await;
        verify(&s, id, &spec, &view, 8);
        assert_eq!(s.log.results_for(id).len(), 8);
    }
    assert_eq!(s.shutdown(2000).await, 0);
    assert_eq!(s.stderr().matches("CUDA lease topology mode").count(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn disabled_screen_forces_direct_even_for_an_unreachable_target() {
    let mut s = start("[screen]\nenabled = false").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(-128_001, 32)))
        .await;
    let job = lease(b"direct", 3, RING_A, 0, 8);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    finish(&mut s, b"direct").await;
    verify(&s, b"direct", &spec, &view, 8);
    assert_eq!(
        s.log.results_for(b"direct").len(),
        8,
        "push ignores the target"
    );
    assert_eq!(s.shutdown(2000).await, 0);
    assert!(s.stderr().contains("Direct") && s.stderr().contains("screen disabled"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn weighted_couplings_take_direct_and_keep_exact_energies() {
    let mut s = start("").await;
    let (mut topology, _) = ring(64, RING_A);
    topology.allowed_j_milli = vec![-1375, 250, 1001];
    let view = TopologyView::from_proto(&topology).unwrap();
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    let job = lease(b"weighted", 3, RING_A, 0, 8);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    finish(&mut s, b"weighted").await;
    verify(&s, b"weighted", &spec, &view, 8);
    assert_eq!(s.log.results_for(b"weighted").len(), 8);
    assert_eq!(s.shutdown(2000).await, 0);
    assert!(s.stderr().contains("Direct") && s.stderr().contains("screen couplings"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA device 0"]
async fn two_leases_back_to_back_finish_in_fifo_order() {
    let mut s = start("").await;
    let (topology, view) = ring(1024, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 16_384)))
        .await;
    let first = lease(b"first", 3, RING_A, 0, 1_000_000);
    let spec_a = generator(&first);
    s.send(coord_msg::Msg::Job(first)).await;
    // A first result proves the first lease owns the turn before the second joins.
    s.until("first turn", LONG, |log| {
        !log.results_for(b"first").is_empty()
    })
    .await;
    let second = lease(b"second", 4, RING_A, 100, 8);
    let spec_b = generator(&second);
    s.send(coord_msg::Msg::Job(second)).await;
    s.until_stderr("CUDA lease waiting for turn", LONG).await;
    assert!(
        s.log.done(b"first").is_none(),
        "the first turn is still held"
    );
    assert!(
        s.log.results_for(b"second").is_empty(),
        "the second lease waits"
    );
    s.send(coord_msg::Msg::Cancel(Cancel { max_generation: 3 }))
        .await;
    finish(&mut s, b"second").await;
    let first_count = s.log.done(b"first").expect("first summary").salts_done;
    assert!(first_count > 0 && first_count < 1_000_000);
    verify(&s, b"first", &spec_a, &view, first_count);
    verify(&s, b"second", &spec_b, &view, 8);
    let first_tail = s
        .log
        .order
        .iter()
        .rposition(|e| e == "result:first")
        .unwrap();
    let second_head = s
        .log
        .order
        .iter()
        .position(|e| e == "result:second")
        .unwrap();
    assert!(first_tail < second_head);
    assert_eq!(s.shutdown(2000).await, 0);
    assert_eq!(s.stderr().matches("CUDA lease topology mode").count(), 1);
}

//! Session and `--solve` conformance: spawn SA, multi-spin SA and Gibbs solvers
//! against quip-solver-conformance's drivers.
//!
//! GPU-backed drive tests are `#[ignore]` so default `cargo test` on headless CI
//! never confuses "self-skipped" with "passed" ([quip-miner-cuda-gp2] part b).
//! Run them on a GPU host: `cargo test -p quip-solver-cuda -- --ignored`.

use quip_solver_conformance::driver::{
    drive_miner, DriverReport, Terminal, CONFIGURED_SWEEPS, GIBBS_SWEEP_MULTIPLIER,
};
use quip_solver_conformance::solve::drive_solve;
use quip_solver_core::quip_proto::v1::RejectReason;
use quip_solver_core::quip_protocol::session::PROTOCOL_VERSION;
use serial_test::serial;
use std::process::Command;

mod common;
use common::{ensure_built, profile_bin};

#[tokio::test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
async fn quip_cuda_sa_passes_conformance() {
    ensure_built(&["quip-cuda-sa"]);
    let miner = profile_bin("quip-cuda-sa");
    let socket = format!(
        "/tmp/quip-cuda-sa-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert!(report.is_conformant(), "{}", report.summary());
}

#[tokio::test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
async fn quip_cuda_sa_passes_solve_conformance() {
    ensure_built(&["quip-cuda-sa"]);
    let miner = profile_bin("quip-cuda-sa");
    let report = drive_solve(&miner).await;
    assert!(report.is_conformant(), "{}", report.summary());
}

#[tokio::test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
async fn quip_cuda_msa_passes_conformance() {
    ensure_built(&["quip-cuda-msa"]);
    let miner = profile_bin("quip-cuda-msa");
    let socket = format!(
        "/tmp/quip-cuda-msa-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert_msa_conformance_with_unit_model_refusals(&report);
}

/// Driver 0.0.3's `valid_ising`, `hash_ising`, `sparse_ising`, and `seeded_ising`
/// fixtures use fractional J or h.
/// Remove this helper when the driver sends unit fixtures to solvers advertising
/// `exact_resolution_milli >= 1000`.
fn assert_msa_conformance_with_unit_model_refusals(report: &DriverReport) {
    assert!(
        report.handshake_ok
            && report.ready_received
            && !report.job_request_credits.is_empty()
            && report.job_request_credits.iter().all(|&c| c > 0)
            && report.has_reject(b"job-bad-h", RejectReason::Malformed)
            && report.has_reject(b"job-bad-j", RejectReason::Malformed)
            && report.has_reject(b"job-gate", RejectReason::UnsupportedKind)
            && report.has_reject(b"job-old", RejectReason::Expired)
            && report.warm_start_conformant()
            && report.lease_conformant()
            && report.cancel_acked
            && report.ping_acked
            && report.capabilities_conformant()
            && report.credit_ledger_balanced()
            && report.terminal == Terminal::Closed
            && report.timed_out_phases.is_empty()
            && report.exit_code == 0,
        "{}",
        report.summary()
    );
    for id in [
        b"job-1".as_slice(),
        b"job-2",
        b"job-hash",
        b"job-sparse",
        b"job-seeded",
        b"job-cancel",
        b"job-stale",
    ] {
        // SampleError::Capacity maps to TooLarge on the 0.0.3 wire.
        assert!(
            report.has_reject(id, RejectReason::TooLarge),
            "{id:?} must be rejected as Capacity: {report:?}"
        );
        assert!(
            !report.results.iter().any(|r| r.job_id == id),
            "{id:?} must not produce a Result: {report:?}"
        );
    }
    assert!(report.energies_rescore_clean(), "{report:?}");
    assert!(report.sweeps_honoured(), "{report:?}");
    assert!(report.cancelled_watermark > 0, "{report:?}");
    assert!(
        report.abandoned_watermark() >= report.cancelled_watermark,
        "{report:?}"
    );
}

#[tokio::test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
async fn quip_cuda_msa_passes_solve_conformance() {
    ensure_built(&["quip-cuda-msa"]);
    let miner = profile_bin("quip-cuda-msa");
    let report = drive_solve(&miner).await;
    assert!(report.is_conformant(), "{}", report.summary());
}

#[tokio::test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
async fn quip_cuda_gibbs_passes_conformance() {
    ensure_built(&["quip-cuda-gibbs"]);
    let miner = profile_bin("quip-cuda-gibbs");
    let socket = format!(
        "/tmp/quip-cuda-gibbs-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    // Algorithm-aware since quip-solver-conformance 0.0.1-rc1: the driver
    // derives the doubled gibbs expectation from the Hello, so the composite
    // verdict grades the doubling itself. Pin the derivation so a miner
    // advertising the wrong algorithm cannot make both sides agree.
    assert_eq!(
        report.expected_meta_sweeps(),
        CONFIGURED_SWEEPS * GIBBS_SWEEP_MULTIPLIER,
        "driver did not derive the gibbs sweep expectation from the Hello"
    );
    assert!(report.is_conformant(), "{}", report.summary());
}

#[tokio::test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
async fn quip_cuda_gibbs_passes_solve_conformance() {
    ensure_built(&["quip-cuda-gibbs"]);
    let miner = profile_bin("quip-cuda-gibbs");
    let report = drive_solve(&miner).await;
    assert!(report.is_conformant(), "{}", report.summary());
}

/// `--capabilities` / `--version` are headless (no CUDA).
#[test]
#[serial]
fn capabilities_and_version_headless() {
    ensure_built(&["quip-cuda-sa", "quip-cuda-msa", "quip-cuda-gibbs"]);

    for (bin, algo) in [
        ("quip-cuda-sa", "sa"),
        ("quip-cuda-msa", "msa"),
        ("quip-cuda-gibbs", "gibbs"),
    ] {
        let path = profile_bin(bin);

        let out = Command::new(&path).arg("--capabilities").output().unwrap();
        assert!(out.status.success(), "{bin} --capabilities failed");
        let s = String::from_utf8(out.stdout).unwrap();
        assert!(s.contains("\"backend\":\"cuda\""), "{bin}: {s}");
        assert!(
            s.contains(&format!("\"algorithm\":\"{algo}\"")),
            "{bin}: {s}"
        );

        let out = Command::new(&path).arg("--version").output().unwrap();
        assert!(out.status.success());
        // The literal in each binary's `#[command(version)]` must track the
        // protocol the linked quip-solver-core speaks.
        let version = String::from_utf8(out.stdout).unwrap();
        assert!(
            version.contains(&format!("protocol {PROTOCOL_VERSION}")),
            "{bin}: {version}"
        );
    }
}

/// `--check` opens the GPU and compiles kernels.
#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
fn check_succeeds_with_gpu() {
    ensure_built(&["quip-cuda-sa", "quip-cuda-msa", "quip-cuda-gibbs"]);

    for bin in ["quip-cuda-sa", "quip-cuda-msa", "quip-cuda-gibbs"] {
        let path = profile_bin(bin);
        let status = Command::new(&path)
            .arg("--check")
            .arg("--device")
            .arg("0")
            .status();
        assert!(
            status.unwrap().success(),
            "{bin} --check must succeed when a GPU is present"
        );
    }
}

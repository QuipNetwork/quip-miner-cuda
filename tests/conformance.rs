//! Protocol conformance: spawn SA, multi-spin SA and Gibbs miners against
//! quip-solver-conformance's driver.
//!
//! GPU-backed drive tests are `#[ignore]` so default `cargo test` on headless CI
//! never confuses "self-skipped" with "passed" ([quip-miner-cuda-gp2] part b).
//! Run them on a GPU host: `cargo test -p quip-solver-cuda -- --ignored`.

use quip_solver_conformance::driver::{drive_miner, CONFIGURED_SWEEPS, GIBBS_SWEEP_MULTIPLIER};
use quip_solver_conformance::solve::{drive_solve, CaseOutcome};
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

/// Every kernel passes the `--solve` gate, runs a linear schedule, and honours
/// the limits it advertises: a model at each limit solves, one past it is
/// refused as capacity.
#[tokio::test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
async fn every_kernel_passes_the_solve_gate() {
    ensure_built(&["quip-cuda-sa", "quip-cuda-msa", "quip-cuda-gibbs"]);
    for (bin, limit_cases) in [
        (
            "quip-cuda-sa",
            &["field-limit-inside", "field-limit-outside"][..],
        ),
        (
            "quip-cuda-msa",
            &[
                "degree-limit-inside",
                "degree-limit-outside",
                "field-limit-inside",
                "field-limit-outside",
            ][..],
        ),
        (
            "quip-cuda-gibbs",
            &["field-limit-inside", "field-limit-outside"][..],
        ),
    ] {
        let report = drive_solve(&profile_bin(bin)).await;
        assert!(report.is_conformant(), "{bin}:\n{}", report.summary());
        for case in ["linear-schedule"].iter().chain(limit_cases) {
            assert_eq!(
                report.outcome(case),
                Some(&CaseOutcome::Passed),
                "{bin} {case}:\n{}",
                report.summary()
            );
        }
    }
}

/// Each kernel lists `linear-schedule` and states its model limits, without a
/// device.
#[test]
#[serial]
fn capabilities_advertise_the_schedule_feature_and_limits() {
    ensure_built(&["quip-cuda-sa", "quip-cuda-msa", "quip-cuda-gibbs"]);
    for (bin, degree, field) in [
        ("quip-cuda-sa", 0, 63_000),
        ("quip-cuda-msa", 20, 21_000),
        ("quip-cuda-gibbs", 0, 127_000),
    ] {
        let out = Command::new(profile_bin(bin))
            .arg("--capabilities")
            .output()
            .unwrap();
        let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert!(
            caps["features"]
                .as_array()
                .is_some_and(|f| f.iter().any(|f| f == "linear-schedule")),
            "{bin}: {caps}"
        );
        assert_eq!(caps["maxDegree"], degree, "{bin}: {caps}");
        assert_eq!(caps["maxLocalFieldMilli"], field, "{bin}: {caps}");
        assert_eq!(caps["exactResolutionMilli"], 1000, "{bin}: {caps}");
    }
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

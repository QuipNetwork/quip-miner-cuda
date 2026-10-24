//! Session and `--solve` conformance: spawn SA, multi-spin SA and Gibbs solvers
//! against quip-solver-conformance's drivers.
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

/// Run the `--solve` gate on `bin` and require, beyond overall conformance,
/// that the linear-schedule case and each named limit case passed rather than
/// being skipped: the gate skips a case the binary does not advertise.
async fn assert_passes_solve_gate(bin: &str, limit_cases: &[&str]) {
    ensure_built(&[bin]);
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
    assert_passes_solve_gate(
        "quip-cuda-sa",
        &["field-limit-inside", "field-limit-outside"],
    )
    .await;
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
    let lease = report.lease.as_ref().expect("MSA advertises salt leases");
    assert_eq!(
        lease.results_verified, 4,
        "the nonzero-field lease uses Direct"
    );
    assert_eq!(lease.lease_done.map(|(salts, _)| salts), Some(4));
    assert!(lease.credit_refunded);
}

#[tokio::test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
async fn quip_cuda_msa_passes_solve_conformance() {
    assert_passes_solve_gate(
        "quip-cuda-msa",
        &["field-limit-inside", "field-limit-outside"],
    )
    .await;
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
    // Gibbs states no field limit (it refuses per job), so the gate skips the
    // field-limit cases.
    assert_passes_solve_gate("quip-cuda-gibbs", &[]).await;
}

/// SA and Gibbs list `linear-schedule` and their limits without a device.
#[test]
#[serial]
fn capabilities_advertise_the_schedule_feature_and_limits() {
    ensure_built(&["quip-cuda-sa", "quip-cuda-msa", "quip-cuda-gibbs"]);
    for (bin, degree, field) in [
        ("quip-cuda-sa", 0, Some(536_870_910)),
        ("quip-cuda-gibbs", 0, None),
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
        assert_eq!(caps["maxLocalFieldMilli"].as_u64(), field, "{bin}: {caps}");
        assert_eq!(caps["exactResolutionMilli"], 1, "{bin}: {caps}");
    }
}

/// SA and Gibbs capabilities and every version are headless (no CUDA).
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

        if algo != "msa" {
            let out = Command::new(&path).arg("--capabilities").output().unwrap();
            assert!(out.status.success(), "{bin} --capabilities failed");
            let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(caps["backend"], "cuda");
            assert_eq!(caps["algorithm"], algo);
        }

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

/// MSA capabilities use the selected device's memory snapshot.
#[test]
#[serial]
#[ignore = "requires CUDA GPU"]
fn msa_capabilities_report_probed_device_limits() {
    ensure_built(&["quip-cuda-msa"]);
    let out = Command::new(profile_bin("quip-cuda-msa"))
        .arg("--capabilities")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(caps["maxDegree"].is_null());
    let nodes = caps["maxNodes"].as_u64().unwrap();
    let edges = caps["maxEdges"].as_u64().unwrap();

    assert!(nodes > 0 && edges > 0, "{caps}");
    // A complete model at the edge limit still fits the independent node limit.
    assert!(edges <= nodes * (nodes - 1) / 2, "{caps}");
    assert_eq!(caps["maxLocalFieldMilli"], 16_777_215);
    assert_eq!(caps["exactResolutionMilli"], 1);
    assert!(caps["features"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f == "linear-schedule"));

    let out = Command::new(profile_bin("quip-cuda-msa"))
        .args(["--capabilities", "--max-nodes", "8192"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["maxNodes"], 8192);
}

/// An absent bench cap uses the device budget, while explicit caps still apply.
#[test]
#[serial]
#[ignore = "requires CUDA GPU"]
fn msa_bench_without_max_nodes_uses_device_limit() {
    ensure_built(&["quip-cuda-msa"]);
    for cap in [None, Some("5000"), Some("6000")] {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new(profile_bin("quip-cuda-msa"));
        if let Some(cap) = cap {
            command.args(["--max-nodes", cap]);
        }
        let out = command
            .args([
                "bench",
                "run",
                "--nodes",
                "6000",
                "--reads",
                "8",
                "--sweeps",
                "4",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--out",
            ])
            .arg(dir.path())
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        if cap == Some("5000") {
            assert!(!out.status.success(), "explicit cap must reject 6000 nodes");
            assert!(
                stderr.contains("6000") && stderr.contains("5000"),
                "{stderr}"
            );
        } else {
            assert!(out.status.success(), "cap {cap:?}: {stderr}");
            let records = std::fs::read_to_string(dir.path().join("cuda-msa_n6000_s4.jsonl"))
                .expect("bench record above the default node floor");
            assert_eq!(records.lines().count(), 1);
        }
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

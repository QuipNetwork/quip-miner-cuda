//! Protocol conformance: spawn SA and Gibbs miners against quip-solver-conformance's driver.
//!
//! GPU-backed drive tests are `#[ignore]` so default `cargo test` on headless CI
//! never confuses "self-skipped" with "passed" ([quip-miner-cuda-gp2] part b).
//! Run them on a GPU host: `cargo test -p quip-miner-cuda -- --ignored`.

use quip_solver_conformance::driver::{drive_miner, CONFIGURED_SWEEPS, GIBBS_SWEEP_MULTIPLIER};
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

/// `--capabilities` / `--version` are headless (no CUDA).
#[test]
#[serial]
fn capabilities_and_version_headless() {
    ensure_built(&["quip-cuda-sa", "quip-cuda-gibbs"]);

    for (bin, algo) in [("quip-cuda-sa", "sa"), ("quip-cuda-gibbs", "gibbs")] {
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
        assert!(String::from_utf8(out.stdout).unwrap().contains("protocol"));
    }
}

/// `--check` opens the GPU and compiles kernels.
#[test]
#[serial]
#[ignore = "requires CUDA GPU; run with cargo test -- --ignored"]
fn check_succeeds_with_gpu() {
    ensure_built(&["quip-cuda-sa", "quip-cuda-gibbs"]);

    for bin in ["quip-cuda-sa", "quip-cuda-gibbs"] {
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

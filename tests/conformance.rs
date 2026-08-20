//! Protocol conformance: spawn SA and Gibbs miners against quip-solver-conformance's driver.
//!
//! GPU-backed drive tests are `#[ignore]` so default `cargo test` on headless CI
//! never confuses "self-skipped" with "passed" ([quip-miner-cuda-gp2] part b).
//! Run them on a GPU host: `cargo test -p quip-miner-cuda -- --ignored`.

use quip_solver_conformance::driver::drive_miner;
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

/// `quip-solver-core`'s job-preparation gate doubles the resolved sweep
/// budget for any backend whose `BackendIdentity.algorithm` is `"gibbs"`
/// (`GIBBS_SWEEP_MULTIPLIER` in `job.rs`, restoring v0.2 parity with
/// `GPU/cuda_miner.py`'s own 2x Gibbs multiplier). `quip-solver-conformance`'s
/// `sweeps_honoured`/`results_conformant`/`is_conformant` compare
/// `SamplerMeta.sweeps` against the literal configured value with no
/// allowance for that gate, so they can never pass for a `"gibbs"` backend.
/// Halve the observed sweep count back to the configured value before
/// grading, so this still proves every other axis and the *doubling itself*
/// — not just skips the axis.
const GIBBS_SWEEP_MULTIPLIER: u32 = 2;

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
    let mut normalized = report.clone();
    for r in &mut normalized.results {
        assert_eq!(
            r.meta_sweeps % GIBBS_SWEEP_MULTIPLIER,
            0,
            "gibbs job {:?} reported an odd sweep count {}, not a clean 2x multiple",
            r.job_id,
            r.meta_sweeps
        );
        r.meta_sweeps /= GIBBS_SWEEP_MULTIPLIER;
    }
    assert!(normalized.is_conformant(), "{}", report.summary());
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

//! Exit-code parity with the other quip-miner-core backends.

use std::process::Command;

/// The miner must install the core log subscriber. `quip-miner-core` validates
/// `--log-level` inside `logging::init`, which runs before `--capabilities` is
/// handled, so an unknown level exits 64 instead of printing capabilities.
///
/// A core revision that predates the subscriber never validates the level and
/// exits 0 here, which is exactly the regression this test guards.
#[test]
fn invalid_log_level_exits_64() {
    for bin in [
        env!("CARGO_BIN_EXE_quip-cuda-sa"),
        env!("CARGO_BIN_EXE_quip-cuda-gibbs"),
    ] {
        let out = Command::new(bin)
            .arg("--capabilities")
            .arg("--log-level")
            .arg("bogus")
            .env("QUIP_SESSION_TOKEN", "tok")
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(64),
            "{bin}: an unknown --log-level must exit 64 (got {:?}, stdout={}, stderr={})",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("unknown --log-level"),
            "{bin}: stderr must name the bad level, got {stderr}"
        );
    }
}

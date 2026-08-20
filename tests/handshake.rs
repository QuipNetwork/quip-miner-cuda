//! Exit-code parity with the other quip-solver-core backends.

use std::process::Command;

/// `quip-solver-core`'s `CommonArgs` validates `--log-level` against a fixed
/// list (`trace`, `debug`, `info`, `warn`, `error`) at clap parse time, before
/// `--capabilities` is handled, so an unknown level exits with clap's usage
/// error code (2) instead of printing capabilities.
///
/// A core revision that predates that validator lets an unknown level reach
/// `logging::init` (or, older still, never validates it at all), which is
/// exactly the regression this test guards.
#[test]
fn invalid_log_level_is_rejected_at_parse_time() {
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
            Some(2),
            "{bin}: an unknown --log-level must exit with clap's usage error code 2 \
             (got {:?}, stdout={}, stderr={})",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("--log-level") && stderr.contains("bogus"),
            "{bin}: stderr must name the bad flag and value, got {stderr}"
        );
    }
}

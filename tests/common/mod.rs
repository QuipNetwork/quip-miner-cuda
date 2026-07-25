//! Shared helpers for quip-miner-cuda integration tests.

use std::process::Command;

/// Cross-package binary path (deps/ → profile/ → bin).
pub fn profile_bin(name: &str) -> String {
    let name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let mut p = std::env::current_exe().expect("test exe path");
    p.pop(); // deps/
    p.pop(); // <profile>/
    p.push(&name);
    p.to_string_lossy().into_owned()
}

pub fn ensure_built(package_bins: &[&str]) {
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-miner-cuda"])
        .status()
        .expect("cargo build quip-miner-cuda");
    assert!(status.success(), "failed to build quip-miner-cuda");
    for b in package_bins {
        assert!(
            std::path::Path::new(&profile_bin(b)).exists(),
            "missing binary {b} at {}",
            profile_bin(b)
        );
    }
}

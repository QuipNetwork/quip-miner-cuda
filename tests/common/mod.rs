//! Shared helpers for quip-solver-cuda integration tests.

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
    let mut command = Command::new(env!("CARGO"));
    command.args(["build", "-p", "quip-solver-cuda"]);
    if !cfg!(debug_assertions) {
        command.arg("--release");
    }
    let status = command
        .env("CUDA_VISIBLE_DEVICES", "")
        .status()
        .expect("cargo build quip-solver-cuda");
    assert!(status.success(), "failed to build quip-solver-cuda");
    for b in package_bins {
        assert!(
            std::path::Path::new(&profile_bin(b)).exists(),
            "missing binary {b} at {}",
            profile_bin(b)
        );
    }
}

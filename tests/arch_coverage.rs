//! The support contract: both kernels compile for every architecture in
//! [`SUPPORTED_ARCHS`] *and* the resulting PTX assembles with `ptxas` for
//! that architecture.
//!
//! The ptxas pass is load-bearing, not belt-and-braces: NVRTC will happily
//! emit instructions the target cannot execute (an unguarded `__nanosleep`
//! compiles fine into `sm_52` PTX), and the failure then surfaces as
//! `CUDA_ERROR_INVALID_PTX` at module load in production. `ptxas -arch=sm_N`
//! runs the same validation the driver JIT does, without a GPU.
//!
//! Needs the CUDA 12.9 toolkit (libnvrtc + ptxas) but no device — hence the
//! ignore; CI and `make test-archs` run it inside the CI image with
//! `--include-ignored`.

use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use quip_miner_cuda::cuda_device::SUPPORTED_ARCHS;
use std::io::Write;
use std::process::{Command, Stdio};

const SA_SRC: &str = include_str!("../kernels/sa.cu");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.cu");

/// Compile one kernel for one arch and assemble the PTX for that same arch.
fn compile_and_assemble(name: &str, src: &str, nodes: usize, arch: i32) -> Result<(), String> {
    let opts = CompileOptions {
        use_fast_math: Some(true),
        options: vec![
            format!("-DQUIP_MAX_NODES={nodes}"),
            format!("--gpu-architecture=compute_{arch}"),
        ],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(src, opts)
        .map_err(|e| format!("{name} @ compute_{arch}: NVRTC: {e}"))?;

    let mut child = Command::new("ptxas")
        .args([
            format!("-arch=sm_{arch}"),
            "-o".into(),
            "/dev/null".into(),
            "/dev/stdin".into(),
        ])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{name} @ sm_{arch}: ptxas spawn: {e} (CUDA toolkit on PATH?)"))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(ptx.to_src().as_bytes())
        .map_err(|e| format!("{name} @ sm_{arch}: ptxas stdin: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("{name} @ sm_{arch}: ptxas wait: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{name} @ sm_{arch}: ptxas: {}",
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

#[test]
#[ignore = "needs the CUDA 12.9 toolkit (make test-archs runs it in the CI image)"]
fn every_supported_arch_compiles_and_assembles_both_kernels() {
    let mut failures = Vec::new();
    for &arch in SUPPORTED_ARCHS {
        // Smallest realistic capacity for SA, the compiled-in default for
        // Gibbs — the same shapes `CudaDevice::open_with_nodes` produces.
        for (name, src, nodes) in [("sa", SA_SRC, 512), ("gibbs", GIBBS_SRC, 4800)] {
            if let Err(e) = compile_and_assemble(name, src, nodes, arch) {
                failures.push(e);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "supported-arch validation failed:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the CUDA 12.9 toolkit (make test-archs runs it in the CI image)"]
fn floor_is_real_one_arch_below_fails() {
    // The contract's lower bound is meaningful: one step below the floor the
    // kernels must NOT build (they call `__nanosleep`, sm_70+). If this
    // starts passing, the kernels gained a pre-Volta path and SUPPORTED_ARCHS
    // should be widened deliberately.
    let below = SUPPORTED_ARCHS[0] - 8; // 70 -> 62 (Pascal)
    assert!(
        compile_and_assemble("sa", SA_SRC, 512, below).is_err(),
        "sa unexpectedly built at compute_{below}; revisit SUPPORTED_ARCHS"
    );
}

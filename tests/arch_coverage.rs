//! The support contract: every kernel compiles for every architecture in
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
//! `--include-ignored`. Off that toolkit both tests skip, for the reasons on
//! [`skip_outside_ci`].

use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use quip_solver_cuda::cuda_device::{SA_WIDE_DEFINE, SUPPORTED_ARCHS};
use std::io::Write;
use std::process::{Command, Stdio};

const SA_SRC: &str = include_str!("../kernels/sa.cu");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.cu");
const MSA_SRC: &str = include_str!("../kernels/msa.cu");
const SCREEN_SRC: &str = include_str!("../kernels/screen.cu");

/// Reports whether the caller must return without testing anything, and says
/// so on stderr when it does.
///
/// Both tests here measure what the *toolkit* accepts, so they only carry
/// meaning on the pinned CUDA 12.9 one. A newer toolkit does not merely fail
/// them, it misreports them. CUDA 13 dropped `sm_70` and `sm_72` and renamed
/// `compute_101`, so [`every_supported_arch_compiles_and_assembles_every_kernel`]
/// fails on three architectures while the kernels are fine, and
/// [`floor_is_real_one_arch_below_fails`] passes for the wrong reason — NVRTC
/// rejects `compute_62` as an unknown `-arch` value long before it can reject
/// `__nanosleep`. A wrong answer in either direction is worse than no answer.
///
/// GitLab sets `CI` in every job. The `Makefile` passes it into the CI image,
/// so `make test-archs` and `make test-gpu` count as well. Nothing else does,
/// which is the intent: the toolkit these tests need only exists there.
fn skip_outside_ci(test: &str) -> bool {
    if std::env::var_os("CI").is_some() {
        return false;
    }
    // Straight to fd 2 rather than `eprintln!`, which libtest captures and
    // then discards for a passing test. A skip nobody sees reads exactly like
    // a test that ran and found nothing wrong.
    drop(writeln!(
        std::io::stderr(),
        "SKIP {test}: needs the CUDA 12.9 toolkit. Run `make test-archs`."
    ));
    true
}

/// Compile one kernel for one arch and assemble the PTX for that same arch.
fn compile_and_assemble(
    name: &str,
    src: &str,
    nodes: usize,
    arch: i32,
    defines: &[&str],
) -> Result<(), String> {
    let mut opts = CompileOptions {
        use_fast_math: Some(true),
        options: vec![
            format!("-DQUIP_MAX_NODES={nodes}"),
            format!("--gpu-architecture=compute_{arch}"),
        ],
        ..Default::default()
    };
    opts.options.extend(defines.iter().map(|s| (*s).to_owned()));
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
fn every_supported_arch_compiles_and_assembles_every_kernel() {
    if skip_outside_ci("every_supported_arch_compiles_and_assembles_every_kernel") {
        return;
    }
    let mut failures = Vec::new();
    for &arch in SUPPORTED_ARCHS {
        // Smallest realistic capacity for SA, the compiled-in default for
        // Gibbs — the same shapes `CudaDevice::open_with_nodes` produces.
        for (name, src, nodes, defines) in [
            ("sa", SA_SRC, 512, &[][..]),
            ("gibbs", GIBBS_SRC, 4800, &[][..]),
            (
                "msa",
                MSA_SRC,
                quip_solver_cuda::capacity::MSA_DEFAULT_NODES,
                &[][..],
            ),
            ("sa-wide", SA_SRC, 512, &[SA_WIDE_DEFINE][..]),
            (
                "msa-weighted-i32",
                MSA_SRC,
                quip_solver_cuda::capacity::MSA_DEFAULT_NODES,
                &["-DQUIP_MSA_WEIGHTED", "-DQUIP_MSA_LANE_I32"][..],
            ),
            (
                "msa-weighted-f32",
                MSA_SRC,
                quip_solver_cuda::capacity::MSA_DEFAULT_NODES,
                &["-DQUIP_MSA_WEIGHTED", "-DQUIP_MSA_LANE_F32"][..],
            ),
            (
                "msa-weighted-int8",
                MSA_SRC,
                quip_solver_cuda::capacity::MSA_DEFAULT_NODES,
                &[
                    "-DQUIP_MSA_WEIGHTED",
                    "-DQUIP_MSA_LANE_I32",
                    "-DQUIP_MSA_COEF_I8",
                ][..],
            ),
            (
                "msa-real-f32",
                MSA_SRC,
                quip_solver_cuda::capacity::MSA_DEFAULT_NODES,
                &[
                    "-DQUIP_MSA_WEIGHTED",
                    "-DQUIP_MSA_LANE_F32",
                    "-DQUIP_MSA_COEF_F32",
                ][..],
            ),
            // The screen sizes its shared memory per launch; the node define is unused.
            ("screen", SCREEN_SRC, 0, &[][..]),
        ] {
            if let Err(e) = compile_and_assemble(name, src, nodes, arch, defines) {
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
    if skip_outside_ci("floor_is_real_one_arch_below_fails") {
        return;
    }
    // The contract's lower bound is meaningful: one step below the floor the
    // kernels must NOT build (they call `__nanosleep`, sm_70+). If this
    // starts passing, the kernels gained a pre-Volta path and SUPPORTED_ARCHS
    // should be widened deliberately.
    let below = SUPPORTED_ARCHS[0] - 8; // 70 -> 62 (Pascal)
    assert!(
        compile_and_assemble("sa", SA_SRC, 512, below, &[]).is_err(),
        "sa unexpectedly built at compute_{below}; revisit SUPPORTED_ARCHS"
    );
}

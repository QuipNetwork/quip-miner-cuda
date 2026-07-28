//! On-disk cache for NVRTC-compiled kernel PTX.
//!
//! NVRTC compiling `kernels/sa.cu` / `kernels/gibbs.cu` to PTX costs tens of
//! seconds at every process start, which distorts throughput/timing
//! measurements. The compile is topology-independent — the kernels take the
//! graph as runtime device buffers, nothing per-topology is baked in — so one
//! cached PTX per (source, GPU arch, driver/NVRTC version) serves every
//! topology and model.
//!
//! [`load_or_compile`] loads the module from cached PTX when a matching entry
//! exists and recompiles (then rewrites the cache) otherwise. A corrupt or
//! incompatible cache file never hard-fails: the module-load error is caught
//! and the kernel is recompiled from source.

use crate::cuda_device::CudaError;
use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::Ptx;
use std::collections::hash_map::DefaultHasher;
use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// Marker for the NVRTC flag set the PTX was produced with. Folded into the
/// cache key so a change to the compile flags invalidates every stored entry.
/// `compile_with_fallback` always passes `use_fast_math`; the fallback's
/// `--gpu-architecture=compute_N` is a deterministic function of the driver
/// version, which is already part of the key.
const FLAGS_TAG: &str = "ffm";

/// Resolve the base cache directory from the given environment values, without
/// touching the real process environment (so it is unit-testable).
///
/// Precedence: `QUIP_CUDA_CACHE` (used verbatim) → `XDG_CACHE_HOME` +
/// `/quip-miner-cuda` → `HOME` + `/.cache/quip-miner-cuda`. Returns `None` when
/// none resolve to a non-empty path.
fn resolve_cache_dir(
    quip: Option<OsString>,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Option<PathBuf> {
    fn non_empty(v: Option<OsString>) -> Option<PathBuf> {
        v.filter(|s| !s.is_empty()).map(PathBuf::from)
    }
    if let Some(dir) = non_empty(quip) {
        return Some(dir);
    }
    if let Some(dir) = non_empty(xdg) {
        return Some(dir.join("quip-miner-cuda"));
    }
    non_empty(home).map(|h| h.join(".cache").join("quip-miner-cuda"))
}

/// Resolve and create the cache directory, reading the real environment.
/// Returns `None` when the directory cannot be resolved or created (caching is
/// then skipped, never fatal).
fn cache_dir() -> Option<PathBuf> {
    let dir = resolve_cache_dir(
        std::env::var_os("QUIP_CUDA_CACHE"),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )?;
    match std::fs::create_dir_all(&dir) {
        Ok(()) => Some(dir),
        Err(e) => {
            warn!("cuda jit cache: cannot create {}: {e}", dir.display());
            None
        }
    }
}

/// `QUIP_CUDA_CACHE_DISABLE` escape hatch: force recompile (skip read + write).
fn cache_disabled() -> bool {
    std::env::var("QUIP_CUDA_CACHE_DISABLE")
        .ok()
        .is_some_and(|v| !v.is_empty() && v != "0")
}

/// Cache key / filename stem for one kernel. Same `(src, arch, driver_version)`
/// yields the same key; a different `src` yields a different key. `arch` and
/// `driver_version` appear verbatim in the stem (readable + they partition the
/// cache); the source text and flag set are folded into the trailing hash.
fn cache_key(kernel: &str, src: &str, arch: &str, driver_version: i32) -> String {
    let mut h = DefaultHasher::new();
    src.hash(&mut h);
    FLAGS_TAG.hash(&mut h);
    let src_hash = h.finish();
    format!("{kernel}-{arch}-drv{driver_version}-{src_hash:016x}")
}

/// Write `text` to `<dir>/<key>.ptx` atomically: write a pid-scoped temp file,
/// then rename it into place so concurrent processes never observe a partial
/// file. Non-fatal — a write failure only forgoes the cache.
fn write_atomic(dir: &std::path::Path, key: &str, text: &str) {
    let final_path = dir.join(format!("{key}.ptx"));
    let tmp = dir.join(format!("{key}.ptx.tmp.{}", std::process::id()));
    if let Err(e) = std::fs::write(&tmp, text) {
        warn!("cuda jit cache: write {} failed: {e}", tmp.display());
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &final_path) {
        warn!(
            "cuda jit cache: rename into {} failed: {e}",
            final_path.display()
        );
        drop(std::fs::remove_file(&tmp));
    }
}

/// Load a kernel module from cached PTX, or compile it and cache the result.
///
/// Behaviour:
/// 1. If caching is enabled and `<dir>/<key>.ptx` loads into a module, use it.
/// 2. Otherwise compile via `compile` (NVRTC), load the module, and write the
///    PTX to the cache atomically.
/// 3. A cached file that fails to load (corrupt / incompatible) is not fatal:
///    the kernel is recompiled and the cache overwritten.
///
/// # Errors
///
/// [`CudaError`] only from `compile` (NVRTC) or the module load of freshly
/// compiled PTX — never from cache I/O, which degrades to a recompile.
pub(crate) fn load_or_compile<F>(
    ctx: &Arc<CudaContext>,
    kernel: &str,
    src: &str,
    arch: &str,
    driver_version: i32,
    compile: F,
) -> Result<Arc<CudaModule>, CudaError>
where
    F: FnOnce() -> Result<Ptx, CudaError>,
{
    let key = cache_key(kernel, src, arch, driver_version);
    let dir = if cache_disabled() { None } else { cache_dir() };

    if let Some(dir) = &dir {
        let path = dir.join(format!("{key}.ptx"));
        if let Ok(text) = std::fs::read_to_string(&path) {
            match ctx.load_module(Ptx::from_src(text)) {
                Ok(module) => {
                    info!("cuda kernel {kernel}: loaded cached PTX ({key})");
                    return Ok(module);
                }
                Err(e) => {
                    warn!("cuda kernel {kernel}: cached PTX unusable ({e}); recompiling");
                }
            }
        }
    }

    let start = Instant::now();
    let ptx = compile()?;
    // `to_src()` borrows; capture the PTX text before `load_module` consumes it.
    let text = ptx.to_src();
    let module = ctx.load_module(ptx)?;
    let ms = start.elapsed().as_millis();

    if let Some(dir) = &dir {
        write_atomic(dir, &key, &text);
        info!("cuda kernel {kernel}: compiled + cached PTX ({key}) in {ms}ms");
    } else {
        info!("cuda kernel {kernel}: compiled PTX ({key}) in {ms}ms (cache off)");
    }
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::{cache_key, resolve_cache_dir};
    use std::path::PathBuf;

    #[test]
    fn key_is_stable_for_same_inputs() {
        let a = cache_key("sa", "__global__ void k(){}", "sm_86", 12080);
        let b = cache_key("sa", "__global__ void k(){}", "sm_86", 12080);
        assert_eq!(a, b);
    }

    #[test]
    fn key_changes_with_source() {
        let a = cache_key("sa", "source one", "sm_86", 12080);
        let b = cache_key("sa", "source two", "sm_86", 12080);
        assert_ne!(a, b);
    }

    #[test]
    fn key_changes_with_arch_and_driver() {
        let base = cache_key("sa", "src", "sm_86", 12080);
        assert_ne!(base, cache_key("sa", "src", "sm_90", 12080));
        assert_ne!(base, cache_key("sa", "src", "sm_86", 12090));
        // Different kernel name → different key even for identical source.
        assert_ne!(base, cache_key("gibbs", "src", "sm_86", 12080));
    }

    #[test]
    fn quip_cuda_cache_overrides_everything() {
        let dir = resolve_cache_dir(
            Some("/explicit/cache".into()),
            Some("/xdg".into()),
            Some("/home/user".into()),
        );
        // QUIP_CUDA_CACHE is used verbatim, no subdirectory appended.
        assert_eq!(dir, Some(PathBuf::from("/explicit/cache")));
    }

    #[test]
    fn falls_back_to_xdg_then_home() {
        assert_eq!(
            resolve_cache_dir(None, Some("/xdg".into()), Some("/home/user".into())),
            Some(PathBuf::from("/xdg/quip-miner-cuda")),
        );
        assert_eq!(
            resolve_cache_dir(None, None, Some("/home/user".into())),
            Some(PathBuf::from("/home/user/.cache/quip-miner-cuda")),
        );
    }

    #[test]
    fn empty_values_are_skipped_and_none_when_unresolvable() {
        // An empty QUIP_CUDA_CACHE is ignored in favour of XDG.
        assert_eq!(
            resolve_cache_dir(Some("".into()), Some("/xdg".into()), None),
            Some(PathBuf::from("/xdg/quip-miner-cuda")),
        );
        assert_eq!(resolve_cache_dir(None, None, None), None);
    }
}

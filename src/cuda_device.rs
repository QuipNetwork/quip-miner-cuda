//! CUDA context + NVRTC-compiled self-feeding kernels for one physical GPU.
//!
//! One process owns one device (`[cuda.N]` → device N / miner id `cuda-N`).

use crate::jit_cache;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;
use tracing::trace_span;

const SA_SRC: &str = include_str!("../kernels/sa.cu");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.cu");

/// Minimum CUDA driver version (`cuDriverGetVersion` encoding: `major*1000 +
/// minor*10`) NVRTC needs to natively target each GPU arch. Port of
/// `GPU/base_cuda_sampler.py::_CUDA_ARCH_MIN_VERSION`.
const CUDA_ARCH_MIN_VERSION: &[(i32, i32)] = &[
    (121, 12090),
    (120, 12080),
    (103, 12090),
    (101, 12080),
    (100, 12080),
    (90, 12000),
    (89, 11080),
    (86, 11010),
    (80, 11000),
];

/// Highest GPU arch the given driver version supports. Port of
/// `_best_fallback_arch`.
fn best_fallback_arch(driver_version: i32) -> i32 {
    CUDA_ARCH_MIN_VERSION
        .iter()
        .filter(|&&(_, min)| min <= driver_version)
        .map(|&(arch, _)| arch)
        .max()
        .unwrap_or(80)
}

/// `cuDriverGetVersion`, wrapped safely (cudarc exposes only the raw sys fn).
fn driver_version() -> Result<i32, CudaError> {
    let mut v: std::ffi::c_int = 0;
    // SAFETY: `v` is a live, initialized `c_int` owned by this frame, so
    // `from_mut(&mut v)` is a valid, aligned, uniquely-borrowed pointer for the
    // whole call — nothing else can alias it. `cuDriverGetVersion` only writes
    // the version through the pointer and does not read or retain it past
    // return, so no obligation outlives this statement.
    unsafe { cudarc::driver::sys::cuDriverGetVersion(std::ptr::from_mut(&mut v)) }.result()?;
    Ok(v)
}

/// Failures from opening a device or compiling its kernels.
#[derive(Debug, Error)]
pub enum CudaError {
    /// A CUDA driver call failed; the payload is the driver's own message.
    #[error("CUDA driver: {0}")]
    Driver(String),
    /// NVRTC rejected the kernel source on both the default and the
    /// architecture-fallback compile.
    #[error("NVRTC compile: {0}")]
    Compile(String),
    /// `device_index` is past the number of devices visible to this process.
    #[error("no CUDA device at index {0}")]
    NoDevice(usize),
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        CudaError::Driver(e.to_string())
    }
}

/// Compile CUDA source with NVRTC, retrying with an explicit
/// `--gpu-architecture=compute_N` PTX fallback if the default (portable,
/// arch-unspecified) compile fails — e.g. a GPU newer than this NVRTC knows
/// natively. The fallback arch is the highest one the installed driver
/// supports; the driver JIT-compiles that PTX up to the real SM at module
/// load time. Port of `_compile_module`.
fn compile_with_fallback(src: &str) -> Result<Ptx, CudaError> {
    let base = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    match compile_ptx_with_opts(src, base) {
        Ok(ptx) => Ok(ptx),
        Err(first_err) => {
            let ver = driver_version()?;
            let fb = best_fallback_arch(ver);
            let opts = CompileOptions {
                use_fast_math: Some(true),
                options: vec![format!("--gpu-architecture=compute_{fb}")],
                ..Default::default()
            };
            compile_ptx_with_opts(src, opts).map_err(|e| {
                CudaError::Compile(format!(
                    "default compile failed ({first_err}); compute_{fb} fallback also failed: {e}"
                ))
            })
        }
    }
}

/// Loaded kernels + streams bound to a single device.
///
/// Every handle field is `pub(crate)`: `open` switches cudarc's per-`CudaSlice`
/// use-after-free event tracking *off* for this context, and the invariant that
/// replaces it (teardown only after `signal_exit` + `synchronize`) can only be
/// upheld by `streaming`/`sampler` inside this crate. Handing any of these out
/// would let a downstream caller allocate against a context whose protection
/// was silently withdrawn. The scalars stay `pub` — they carry no capability.
pub struct CudaDevice {
    /// Zero-based index of the physical GPU this device was opened on.
    pub device_index: usize,
    pub(crate) ctx: Arc<CudaContext>,
    /// The device's default (null) stream. `streaming` builds its own
    /// compute/transfer streams, so nothing reads this today.
    #[allow(dead_code)]
    pub(crate) stream: Arc<CudaStream>,
    /// `cuda_sa_self_feeding` — persistent kernel, 1 block (1 SM) per nonce.
    pub(crate) sa: CudaFunction,
    /// `cuda_gibbs_self_feeding` — persistent kernel, `sms_per_nonce` blocks
    /// per nonce.
    pub(crate) gibbs: CudaFunction,
    /// SMs on this device (`launch_self_feeding`'s `num_kernels` budget).
    pub max_sms: usize,
    _sa_mod: Arc<CudaModule>,
    _gibbs_mod: Arc<CudaModule>,
}

// Scalar device facts only; the context, stream, kernel handles and loaded
// modules are deliberately omitted (raw CUDA pointers, no diagnostic value).
impl fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDevice")
            .field("device_index", &self.device_index)
            .field("max_sms", &self.max_sms)
            .finish_non_exhaustive()
    }
}

impl CudaDevice {
    /// Create a context on `device_index` and NVRTC-compile the kernels.
    ///
    /// # Errors
    ///
    /// - [`CudaError::NoDevice`] if `device_index` is at or past the number of
    ///   CUDA devices visible to this process.
    /// - [`CudaError::Driver`] on any driver failure: the device-count query,
    ///   context creation, the SM-count attribute query, module load, or
    ///   kernel function load.
    /// - [`CudaError::Compile`] if NVRTC rejects a kernel on both the default
    ///   pass and the `compute_N` arch-fallback pass.
    ///
    /// ```no_run
    /// use quip_miner_cuda::cuda_device::CudaDevice;
    ///
    /// let device = CudaDevice::open(0)?;
    /// println!("device {} has {} SMs", device.device_index, device.max_sms);
    /// # Ok::<(), quip_miner_cuda::cuda_device::CudaError>(())
    /// ```
    pub fn open(device_index: usize) -> Result<Self, CudaError> {
        // CUDA reports counts as i32; reject a negative driver response rather
        // than silent truncation into usize.
        let n = usize::try_from(CudaContext::device_count()?)
            .map_err(|_| CudaError::Driver("CUDA reported a negative device count".into()))?;
        if device_index >= n {
            return Err(CudaError::NoDevice(device_index));
        }
        let ctx = CudaContext::new(device_index)?;

        // The self-feeding streaming session runs a persistent kernel on one
        // stream while a second stream concurrently uploads/downloads slot
        // data the kernel is still reading/writing (by design: the kernel's
        // own volatile ctrl protocol + __threadfence calls are the
        // synchronization, matching the reference CuPy driver's raw async
        // streams). cudarc's default per-CudaSlice read/write event
        // tracking would instead insert a wait for the (never-until-exit
        // signaled) kernel completion event on the transfer stream, which
        // would deadlock the self-feeding protocol. Safety: every buffer the
        // persistent kernel touches is torn down only after `signal_exit` +
        // `stream_compute.synchronize()` (see `streaming::SelfFeedingSession`
        // drop), so no CudaSlice is freed while still in use. That teardown is
        // load-bearing rather than best-effort: if the final synchronize
        // fails, the session's drop must abort rather than free buffers the
        // kernel may still be reading or writing.
        unsafe { ctx.disable_event_tracking() };

        let stream = ctx.default_stream();

        let max_sms = usize::try_from(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )
        .map_err(|_| CudaError::Driver("CUDA reported a negative SM count".into()))?;

        // Cache key components: GPU arch + driver/NVRTC version. The compiled
        // PTX is topology-independent, so these plus the source text fully
        // determine a cache entry (see `jit_cache`).
        let cc_major =
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let cc_minor =
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let arch = format!("sm_{cc_major}{cc_minor}");
        let driver_ver = driver_version()?;

        let (sa_mod, gibbs_mod) = {
            let _span = trace_span!("jit", kernels = 2).entered();
            (
                jit_cache::load_or_compile(&ctx, "sa", SA_SRC, &arch, driver_ver, || {
                    compile_with_fallback(SA_SRC)
                })?,
                jit_cache::load_or_compile(&ctx, "gibbs", GIBBS_SRC, &arch, driver_ver, || {
                    compile_with_fallback(GIBBS_SRC)
                })?,
            )
        };

        let sa = sa_mod.load_function("cuda_sa_self_feeding")?;
        let gibbs = gibbs_mod.load_function("cuda_gibbs_self_feeding")?;

        Ok(Self {
            device_index,
            ctx,
            stream,
            sa,
            gibbs,
            max_sms: max_sms.max(1),
            _sa_mod: sa_mod,
            _gibbs_mod: gibbs_mod,
        })
    }

    /// Number of CUDA devices visible to this process.
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] if the driver cannot report a device count (no
    /// driver installed, or CUDA failed to initialize).
    ///
    /// ```no_run
    /// use quip_miner_cuda::cuda_device::CudaDevice;
    ///
    /// println!("{} CUDA device(s) visible", CudaDevice::device_count()?);
    /// # Ok::<(), quip_miner_cuda::cuda_device::CudaError>(())
    /// ```
    pub fn device_count() -> Result<usize, CudaError> {
        // CUDA reports the count as i32; reject negative rather than truncate.
        usize::try_from(CudaContext::device_count()?)
            .map_err(|_| CudaError::Driver("CUDA reported a negative device count".into()))
    }

    /// Probe that a device can open and compile kernels (`--check`).
    ///
    /// # Errors
    ///
    /// The same set [`open`](Self::open) reports, since this is a full open
    /// that discards the device: [`CudaError::NoDevice`] for an out-of-range
    /// index, [`CudaError::Driver`] for any driver failure, and
    /// [`CudaError::Compile`] when NVRTC fails both the default and the
    /// arch-fallback compile.
    ///
    /// ```no_run
    /// use quip_miner_cuda::cuda_device::CudaDevice;
    ///
    /// CudaDevice::check(0)?;
    /// # Ok::<(), quip_miner_cuda::cuda_device::CudaError>(())
    /// ```
    pub fn check(device_index: usize) -> Result<(), CudaError> {
        // The probe is the open itself; the device is dropped straight away.
        drop(Self::open(device_index)?);
        Ok(())
    }

    /// The GPU's marketing name (e.g. "NVIDIA H100 80GB HBM3"), for the
    /// `bench` subcommand's `BenchRecord.device` field.
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] if the driver cannot report the device name.
    pub fn name(&self) -> Result<String, CudaError> {
        Ok(self.ctx.name()?)
    }
}

#[cfg(test)]
mod arch_tests {
    use super::best_fallback_arch;

    // `best_fallback_arch` is a pure lookup over CUDA_ARCH_MIN_VERSION, so
    // these run without a GPU or a CUDA driver.

    #[test]
    fn picks_the_highest_arch_the_driver_supports() {
        // 12079: only the entries with a minimum <= 12079 qualify, the
        // highest of which is sm_90 (min 12000).
        assert_eq!(best_fallback_arch(12079), 90);
        // 12080 unlocks 120/101/100; 120 is the highest of those.
        assert_eq!(best_fallback_arch(12080), 120);
        // 12090 additionally unlocks 121 and 103.
        assert_eq!(best_fallback_arch(12090), 121);
    }

    #[test]
    fn floors_at_sm_80_below_every_table_entry() {
        // No entry has a minimum <= 10000, so the `unwrap_or` floor applies.
        assert_eq!(best_fallback_arch(10000), 80);
        // 11000 is the lowest real entry and lands on the same arch.
        assert_eq!(best_fallback_arch(11000), 80);
    }
}

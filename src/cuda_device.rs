//! CUDA context + NVRTC-compiled self-feeding kernels for one physical GPU.
//!
//! One process owns one device (`[cuda.N]` → device N / miner id `cuda-N`).

use crate::capacity;
use crate::jit_cache;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use quip_miner_core::Algorithm;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;
use tracing::trace_span;

const SA_SRC: &str = include_str!("../kernels/sa.cu");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.cu");

/// Every GPU architecture the miner supports: the intersection of what NVRTC
/// 12.9 targets natively (`sm_50..sm_121`, measured via `nvrtcGetSupportedArchs`
/// 2026-08-14) and what the kernels require — both call `__nanosleep`, an
/// `sm_70+` instruction, so the floor is Volta regardless of toolkit.
///
/// This is the support contract: `tests/arch_coverage.rs` compiles both
/// kernels for each entry and assembles the PTX with `ptxas`, so growing or
/// shrinking this list is a reviewed, CI-checked decision rather than a side
/// effect of a toolkit bump.
pub const SUPPORTED_ARCHS: &[i32] = &[
    70, 72, // Volta
    75, // Turing
    80, 86, 87, 89, // Ampere / Ada
    90, // Hopper
    100, 101, 103, // Blackwell datacenter
    120, 121, // Blackwell consumer
];

/// Highest supported arch at or below the detected compute capability.
///
/// Above the ceiling clamps to 121 (PTX loads forward through the driver
/// JIT); below the floor clamps to 70, where the load then fails at the
/// driver with a clear per-device error instead of inside NVRTC; a gap value
/// (e.g. 88, which only CUDA 13 tables know) selects the next lower entry.
///
/// This replaces a driver-version fallback table that could select an arch
/// *newer* than the actual device (r610 driver → `compute_121` on an `sm_86`
/// card) — and PTX only loads forward, so that open failed with
/// `CUDA_ERROR_INVALID_PTX`.
fn select_arch(cc: i32) -> i32 {
    SUPPORTED_ARCHS
        .iter()
        .copied()
        .filter(|&a| a <= cc)
        .max()
        .unwrap_or(70)
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
    /// A requested node capacity was refused before any compile.
    #[error("node capacity: {0}")]
    Capacity(#[from] crate::capacity::CapacityError),
    /// NVRTC rejected the kernel source for the selected architecture.
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

/// Compile CUDA source with NVRTC for one specific architecture.
///
/// `arch` must come from [`select_arch`], which only emits values NVRTC 12.9
/// supports — so a failure here is a real kernel error and there is no
/// fallback pass. The arch-unspecified compile the fallback served no longer
/// exists: NVRTC 12.9's default target (`sm_52`) predates the kernels'
/// `__nanosleep` floor, so a portable compile can never succeed.
fn compile_for_arch(src: &str, max_nodes: usize, arch: i32) -> Result<Ptx, CudaError> {
    // QUIP_MAX_NODES sizes the kernel's state array and must match the
    // jit_cache key's max_nodes component (see `jit_cache`).
    let opts = CompileOptions {
        use_fast_math: Some(true),
        options: vec![
            format!("-DQUIP_MAX_NODES={max_nodes}"),
            format!("--gpu-architecture=compute_{arch}"),
        ],
        ..Default::default()
    };
    compile_ptx_with_opts(src, opts)
        .map_err(|e| CudaError::Compile(format!("compute_{arch} compile failed: {e}")))
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
    /// Node capacity the running algorithm's kernel was compiled for.
    pub max_nodes: usize,
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
            .field("max_nodes", &self.max_nodes)
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
    /// - [`CudaError::Compile`] if NVRTC rejects a kernel for the selected
    ///   `compute_N` architecture.
    ///
    /// ```no_run
    /// use quip_miner_cuda::cuda_device::CudaDevice;
    ///
    /// let device = CudaDevice::open(0)?;
    /// println!("device {} has {} SMs", device.device_index, device.max_sms);
    /// # Ok::<(), quip_miner_cuda::cuda_device::CudaError>(())
    /// ```
    pub fn open(device_index: usize) -> Result<Self, CudaError> {
        Self::open_with_nodes(device_index, Algorithm::Sa, capacity::SA_DEFAULT_NODES)
    }

    /// [`CudaDevice::open`], compiling `algorithm`'s kernel for `max_nodes`.
    ///
    /// The other algorithm's kernel compiles at its own default. One process
    /// drives one algorithm, so only `algorithm` needs the larger array, and a
    /// bigger SA array would cost local memory for a kernel that never
    /// launches here.
    ///
    /// # Errors
    ///
    /// Everything [`CudaDevice::open`] returns, plus [`CudaError::Capacity`]
    /// when `max_nodes` exceeds the algorithm bound or this device's
    /// shared-memory budget.
    pub fn open_with_nodes(
        device_index: usize,
        algorithm: Algorithm,
        max_nodes: usize,
    ) -> Result<Self, CudaError> {
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

        // Node capacity is a `-D` on the kernel, so it is resolved before the
        // compile and must reach both `compile_for_arch` and the cache
        // key. Gibbs is bounded by this device's shared memory, so the budget
        // comes from the device rather than a constant.
        let shared_per_block = usize::try_from(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
        )
        .map_err(|_| CudaError::Driver("CUDA reported negative shared memory".into()))?;
        let threads_per_sm = usize::try_from(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
        )
        .map_err(|_| CudaError::Driver("CUDA reported negative threads per SM".into()))?;
        // Free rather than total: the SA local-memory reservation competes
        // with whatever else already holds memory on this device, and a
        // budget derived from total would promise capacity that is not there.
        let (free_bytes, _total_bytes) = ctx.mem_get_info()?;
        let limits = capacity::DeviceLimits {
            shared_bytes_per_block: shared_per_block,
            free_bytes,
            sm_count: max_sms.max(1),
            threads_per_sm,
        };
        let resolved = capacity::resolve(algorithm, max_nodes, &limits)?;
        let (sa_nodes, gibbs_nodes) = match algorithm {
            Algorithm::Sa => (resolved, capacity::GIBBS_DEFAULT_NODES),
            Algorithm::Gibbs => (capacity::SA_DEFAULT_NODES, resolved),
        };

        // Detected capability -> clamped compile target. An unreadable
        // attribute degrades to the floor (70) rather than failing the open:
        // compute_70 PTX loads on every supported card. The selected arch
        // feeds both the compile and the cache key, so the key always
        // describes the artifact it stores.
        let cc = match (
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        ) {
            (Ok(major), Ok(minor)) => major * 10 + minor,
            _ => 70,
        };
        let sel = select_arch(cc);
        let arch = format!("sm_{sel}");
        let driver_ver = driver_version()?;

        let (sa_mod, gibbs_mod) = {
            let _span = trace_span!("jit", kernels = 2).entered();
            (
                jit_cache::load_or_compile(
                    &ctx,
                    "sa",
                    SA_SRC,
                    &arch,
                    driver_ver,
                    sa_nodes,
                    || compile_for_arch(SA_SRC, sa_nodes, sel),
                )?,
                jit_cache::load_or_compile(
                    &ctx,
                    "gibbs",
                    GIBBS_SRC,
                    &arch,
                    driver_ver,
                    gibbs_nodes,
                    || compile_for_arch(GIBBS_SRC, gibbs_nodes, sel),
                )?,
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
            max_nodes: resolved,
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
    /// [`CudaError::Compile`] when NVRTC rejects a kernel for the selected
    /// architecture.
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
    use super::{select_arch, SUPPORTED_ARCHS};

    // `select_arch` is a pure lookup over SUPPORTED_ARCHS, so these run
    // without a GPU or a CUDA driver.

    #[test]
    fn in_range_value_passes_through() {
        assert_eq!(select_arch(70), 70);
        assert_eq!(select_arch(86), 86);
        assert_eq!(select_arch(121), 121);
    }

    #[test]
    fn newer_than_ceiling_clamps_down_to_121() {
        // A future card past consumer Blackwell: PTX for compute_121 still
        // loads forward through the driver JIT.
        assert_eq!(select_arch(130), 121);
    }

    #[test]
    fn older_than_floor_clamps_up_to_70() {
        // Pascal (61) predates the kernels' `__nanosleep` floor; compute_70
        // is the lowest thing NVRTC can emit for these kernels, and the open
        // then fails at the driver with a clear error instead of inside
        // NVRTC.
        assert_eq!(select_arch(61), 70);
        assert_eq!(select_arch(0), 70);
    }

    #[test]
    fn gap_selects_next_lower_supported_arch() {
        // cc 8.8 exists only in CUDA 13's tables; NVRTC 12.9 cannot target
        // it natively, so emit PTX for the next lower entry (sm_87) and let
        // the driver JIT forward.
        assert_eq!(select_arch(88), 87);
        // cc 9.5 (unknown to any table) falls back to sm_90 the same way.
        assert_eq!(select_arch(95), 90);
    }

    #[test]
    fn table_is_sorted_unique_and_bounded() {
        assert!(SUPPORTED_ARCHS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*SUPPORTED_ARCHS.first().unwrap(), 70);
        assert_eq!(*SUPPORTED_ARCHS.last().unwrap(), 121);
        assert_eq!(SUPPORTED_ARCHS.len(), 13);
    }
}

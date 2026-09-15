//! CUDA context + NVRTC-compiled self-feeding kernels for one physical GPU.
//!
//! One process owns one device (`[cuda.N]` → device N / miner id `cuda-N`).

use crate::capacity;
use crate::jit_cache;
use crate::kernel::KernelKind;
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute_enum};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;
use tracing::trace_span;

const SA_SRC: &str = include_str!("../kernels/sa.cu");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.cu");
/// Multi-spin coded SA, see `kernels/msa.cu`.
const MSA_SRC: &str = include_str!("../kernels/msa.cu");

/// Every GPU architecture the miner supports: the intersection of what NVRTC
/// 12.9 targets natively (`sm_50..sm_121`, measured via `nvrtcGetSupportedArchs`
/// 2026-08-14) and what the kernels require — all three call `__nanosleep`, an
/// `sm_70+` instruction, so the floor is Volta regardless of toolkit.
///
/// This is the support contract: `tests/arch_coverage.rs` compiles every
/// kernel for each entry and assembles the PTX with `ptxas`, so growing or
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

/// PTX modules for the three kernels, in `(sa, msa, gibbs)` order.
type CompiledKernels = (Arc<CudaModule>, Arc<CudaModule>, Arc<CudaModule>);

/// Compile all three kernels for `arch`, keyed by [`KernelKind::name`] in the
/// JIT cache. Kept out of [`CudaDevice::open_with_nodes`] to stay under the
/// crate's function-length cap.
fn compile_all_kernels(
    ctx: &Arc<CudaContext>,
    arch: &str,
    driver_ver: i32,
    sa_nodes: usize,
    msa_nodes: usize,
    gibbs_nodes: usize,
    sel: i32,
) -> Result<CompiledKernels, CudaError> {
    let _span = trace_span!("jit", kernels = 3).entered();
    Ok((
        jit_cache::load_or_compile(
            ctx,
            KernelKind::Sa.name(),
            SA_SRC,
            arch,
            driver_ver,
            sa_nodes,
            || compile_for_arch(SA_SRC, sa_nodes, sel),
        )?,
        jit_cache::load_or_compile(
            ctx,
            KernelKind::Msa.name(),
            MSA_SRC,
            arch,
            driver_ver,
            msa_nodes,
            || compile_for_arch(MSA_SRC, msa_nodes, sel),
        )?,
        jit_cache::load_or_compile(
            ctx,
            KernelKind::Gibbs.name(),
            GIBBS_SRC,
            arch,
            driver_ver,
            gibbs_nodes,
            || compile_for_arch(GIBBS_SRC, gibbs_nodes, sel),
        )?,
    ))
}

/// Verify the loaded msa kernel declares no more static shared
/// memory than [`capacity::MSA_STATIC_SHARED_BYTES`], opt it in to
/// `shared_per_block_optin` less that constant, and return that
/// dynamic-shared-memory budget.
/// Kept out of [`CudaDevice::open_with_nodes`] to stay under the crate's
/// function-length cap.
fn configure_msa_shared_memory(
    msa: &CudaFunction,
    shared_per_block_optin: usize,
) -> Result<usize, CudaError> {
    // The msa kernel sizes its spin state as dynamic shared memory, and a
    // block may only opt in to the device ceiling less the kernel's
    // static members. `capacity::MSA_STATIC_SHARED_BYTES` mirrors those
    // members; check the loaded function declares no more, so the budget the
    // capacity model derived is never larger than the budget the launch gets.
    let static_shared = usize::try_from(msa.shared_size_bytes()?)
        .map_err(|_| CudaError::Driver("CUDA reported negative static shared memory".into()))?;
    if static_shared > capacity::MSA_STATIC_SHARED_BYTES {
        return Err(CudaError::Driver(format!(
            "msa kernel declares {static_shared} bytes of static shared memory, above \
             capacity::MSA_STATIC_SHARED_BYTES ({})",
            capacity::MSA_STATIC_SHARED_BYTES
        )));
    }
    let msa_dynamic_shared_bytes = capacity::msa_dynamic_shared_bytes(shared_per_block_optin);
    let optin = i32::try_from(msa_dynamic_shared_bytes).map_err(|_| {
        CudaError::Driver("opt-in shared memory does not fit the driver's int".into())
    })?;
    msa.set_attribute(
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        optin,
    )?;
    Ok(msa_dynamic_shared_bytes)
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
    /// PCI bus id of that GPU, in NVML's `domain:bus:device.function` form.
    ///
    /// The CUDA ordinal above cannot be used to reach the same GPU through
    /// NVML: NVML enumerates in PCI bus order, while a CUDA ordinal follows
    /// `CUDA_DEVICE_ORDER` (default `FASTEST_FIRST`) and is further remapped
    /// by `CUDA_VISIBLE_DEVICES`. The bus id is the one name both APIs agree
    /// on, so the governor resolves its handle from this.
    pub pci_bus_id: String,
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
    /// `cuda_msa_self_feeding` — persistent kernel, 1 block (1 SM) per
    /// nonce, 64 reads per replica word.
    pub(crate) msa: CudaFunction,
    /// Dynamic shared memory one msa block may request: the device's opt-in
    /// ceiling less the kernel's static members. `streaming` sizes msa
    /// sessions against it; `capacity::msa_budget` derives `max_nodes` from
    /// it when the process runs msa.
    pub(crate) msa_dynamic_shared_bytes: usize,
    /// SMs on this device (`launch_self_feeding`'s `num_kernels` budget).
    pub max_sms: usize,
    /// Node capacity the running kernel was compiled for.
    pub max_nodes: usize,
    _sa_mod: Arc<CudaModule>,
    _gibbs_mod: Arc<CudaModule>,
    _msa_mod: Arc<CudaModule>,
}

// Scalar device facts only; the context, stream, kernel handles and loaded
// modules are deliberately omitted (raw CUDA pointers, no diagnostic value).
impl fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDevice")
            .field("device_index", &self.device_index)
            .field("pci_bus_id", &self.pci_bus_id)
            .field("max_sms", &self.max_sms)
            .field("max_nodes", &self.max_nodes)
            .field("msa_dynamic_shared_bytes", &self.msa_dynamic_shared_bytes)
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
        Self::open_with_nodes(device_index, KernelKind::Sa, capacity::SA_DEFAULT_NODES)
    }

    /// [`CudaDevice::open`], compiling `kernel` for `max_nodes`.
    ///
    /// The other two kernels compile at their own defaults. One process
    /// drives one kernel, so only `kernel` needs the larger array, and a
    /// bigger array would cost memory for a kernel that never launches here.
    ///
    /// # Errors
    ///
    /// Everything [`CudaDevice::open`] returns, plus [`CudaError::Capacity`]
    /// when `max_nodes` exceeds the kernel's bound on this device.
    pub fn open_with_nodes(
        device_index: usize,
        kernel: KernelKind,
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
        let shared_per_block_optin = usize::try_from(ctx.attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )?)
        .map_err(|_| CudaError::Driver("CUDA reported negative opt-in shared memory".into()))?;
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
            shared_bytes_per_block_optin: shared_per_block_optin,
            free_bytes,
            sm_count: max_sms.max(1),
            threads_per_sm,
        };
        let resolved = capacity::resolve(kernel, max_nodes, &limits)?;
        // The selected kernel gets the resolved capacity; the other two
        // compile at their defaults.
        let nodes_for = |k: KernelKind| {
            if k == kernel {
                resolved
            } else {
                capacity::default_nodes(k)
            }
        };
        let sa_nodes = nodes_for(KernelKind::Sa);
        let msa_nodes = nodes_for(KernelKind::Msa);
        let gibbs_nodes = nodes_for(KernelKind::Gibbs);

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

        let (sa_mod, msa_mod, gibbs_mod) = compile_all_kernels(
            &ctx,
            &arch,
            driver_ver,
            sa_nodes,
            msa_nodes,
            gibbs_nodes,
            sel,
        )?;

        let sa = sa_mod.load_function("cuda_sa_self_feeding")?;
        let msa = msa_mod.load_function("cuda_msa_self_feeding")?;
        let gibbs = gibbs_mod.load_function("cuda_gibbs_self_feeding")?;
        let msa_dynamic_shared_bytes = configure_msa_shared_memory(&msa, shared_per_block_optin)?;

        // Read before the struct is built so an unreadable attribute fails the
        // open rather than leaving a device whose governor can never bind.
        let pci_bus_id = format_pci_bus_id(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?,
        );

        Ok(Self {
            device_index,
            pci_bus_id,
            ctx,
            stream,
            sa,
            gibbs,
            msa,
            msa_dynamic_shared_bytes,
            max_sms: max_sms.max(1),
            max_nodes: resolved,
            _sa_mod: sa_mod,
            _gibbs_mod: gibbs_mod,
            _msa_mod: msa_mod,
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

/// NVML's textual PCI bus id, `%08X:%02X:%02X.0`, from the three CUDA device
/// attributes that carry the same address.
///
/// `nvmlDeviceGetHandleByPciBusId` parses this exact shape, and it is what
/// `nvidia-smi` prints, so an operator can match a miner's log line against
/// `nvidia-smi --query-gpu=pci.bus_id` directly. The function is always 0:
/// CUDA reports no function digit, and a GPU is never a multi-function device.
///
/// The three inputs are `i32` because that is what `CudaContext::attribute`
/// returns; a driver reporting a negative address would be a driver bug, and
/// the hex formatting below renders it harmlessly rather than panicking.
#[must_use]
fn format_pci_bus_id(domain: i32, bus: i32, device: i32) -> String {
    format!("{domain:08X}:{bus:02X}:{device:02X}.0")
}

/// The CUDA ordinal a `cuda-N` miner id names, when the label has that shape.
///
/// Deliberately strict — all ASCII digits, no sign, no leading zero except
/// `"0"` — so a free-form label that merely looks device-shaped cannot produce
/// a spurious warning. Returns `None` for any other label (`drive-0`,
/// `mock-0`, `rig7-a`), which imply nothing about the device.
///
/// Advisory only: this never selects a device. The coordinator owns the
/// `[cuda.N]` -> device mapping and passes it as `--device N`.
fn device_from_miner_id(miner_id: &str) -> Option<usize> {
    let digits = miner_id.strip_prefix("cuda-")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // A leading zero ("cuda-01") is a different label from "cuda-1"; treating
    // both as device 1 would let a label the coordinator never emits pass as
    // one it does.
    if digits.len() > 1 && digits.starts_with('0') {
        return None;
    }
    digits.parse().ok()
}

/// The ordinal a `cuda-N` miner id names, when it disagrees with `device`.
///
/// `None` means there is nothing to report: the label names no device, or it
/// names the one `--device` selected. Both binaries decide identically, so the
/// comparison lives here rather than twice in `main`.
///
/// Advisory only. The label never selects a device — overriding `--device`
/// from it would move a running miner to a GPU the operator's config may not
/// have — so a disagreement is a log line, not an error.
#[must_use]
pub fn device_label_mismatch(miner_id: Option<&str>, device: usize) -> Option<usize> {
    let labelled = miner_id.and_then(device_from_miner_id)?;
    (labelled != device).then_some(labelled)
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

#[cfg(test)]
mod pci_bus_id_tests {
    use super::format_pci_bus_id;

    /// The shape `nvmlDeviceGetHandleByPciBusId` parses and `nvidia-smi
    /// --query-gpu=pci.bus_id` prints. Widths are load-bearing: NVML does not
    /// match an unpadded address.
    #[test]
    fn renders_nvml_and_nvidia_smi_shape() {
        assert_eq!(format_pci_bus_id(0, 3, 0), "00000000:03:00.0");
        assert_eq!(format_pci_bus_id(0, 6, 0), "00000000:06:00.0");
    }

    /// Bus and device numbers past 15 must stay two hex digits, not widen.
    #[test]
    fn hex_digits_are_uppercase_and_not_truncated() {
        assert_eq!(format_pci_bus_id(0, 0x1A, 0x0B), "00000000:1A:0B.0");
        assert_eq!(format_pci_bus_id(1, 0xFF, 0xFF), "00000001:FF:FF.0");
    }
}

#[cfg(test)]
mod miner_id_tests {
    use super::{device_from_miner_id, device_label_mismatch};

    // Pure string parsing: no GPU, no CUDA driver.

    #[test]
    fn coordinator_shaped_labels_yield_their_ordinal() {
        assert_eq!(device_from_miner_id("cuda-0"), Some(0));
        assert_eq!(device_from_miner_id("cuda-1"), Some(1));
        assert_eq!(device_from_miner_id("cuda-10"), Some(10));
    }

    /// Anything that is not exactly `cuda-<decimal>` names no device, so the
    /// startup cross-check stays silent instead of warning on a label that
    /// only resembles one.
    #[test]
    fn malformed_cuda_labels_name_no_device() {
        for label in [
            "cuda-", "cuda-x", "cuda-01", "cuda-+1", "cuda- 1", "cuda-1.0", "cuda--1",
        ] {
            assert_eq!(device_from_miner_id(label), None, "{label}");
        }
    }

    /// The conformance driver spawns miners as `--miner-id mock-0` and
    /// `quip-coordinator drive` uses `drive-0`; neither implies a device, so
    /// neither may warn.
    #[test]
    fn foreign_labels_name_no_device() {
        for label in ["drive-0", "mock-0", "rig7-a", ""] {
            assert_eq!(device_from_miner_id(label), None, "{label}");
        }
    }

    /// The bug this diagnostic exists for: the coordinator sends
    /// `--miner-id cuda-1` and no `--device`, so the process labels itself
    /// device 1 while clap's default puts it on device 0.
    #[test]
    fn label_disagreeing_with_the_flag_is_reported() {
        assert_eq!(device_label_mismatch(Some("cuda-1"), 0), Some(1));
        assert_eq!(device_label_mismatch(Some("cuda-0"), 2), Some(0));
    }

    /// Agreement is the supervised path after the coordinator forwards
    /// `--device N`, and it must not log.
    #[test]
    fn label_agreeing_with_the_flag_is_silent() {
        assert_eq!(device_label_mismatch(Some("cuda-0"), 0), None);
        assert_eq!(device_label_mismatch(Some("cuda-3"), 3), None);
    }

    /// A label that names no device implies nothing to contradict, whatever
    /// `--device` says. Covers the `drive-0`/`mock-0` callers and the
    /// pre-default `None` an operator sees before the id is filled in.
    #[test]
    fn labels_naming_no_device_are_silent() {
        assert_eq!(device_label_mismatch(Some("drive-0"), 1), None);
        assert_eq!(device_label_mismatch(Some("mock-0"), 1), None);
        assert_eq!(device_label_mismatch(None, 1), None);
    }
}

//! Single-job sampling entry point.
//!
//! `sample_ising` drives the self-feeding kernel (see [`crate::streaming`])
//! through a dedicated one-nonce session: same kernels, same host-side
//! quantization/coloring as the streaming path, just without the 3-slot
//! rotation across multiple concurrent models. Energies are always scored
//! host-side with [`quip_protocol::scoring::energy_milli`] for consensus;
//! the kernel's own (int8-quantized) energy tracking only drives its
//! internal accept/reject decisions during annealing.

use crate::cuda_device::CudaDevice;
use crate::kernel::KernelKind;
use crate::streaming;
use quip_solver_core::{IsingGraph, SampleParams, SamplerResult};
use thiserror::Error;

/// Failures from running one sampling job on the GPU.
#[derive(Debug, Error)]
pub enum SampleError {
    /// Opening the device or compiling its kernels failed.
    #[error(transparent)]
    Cuda(#[from] crate::cuda_device::CudaError),
    /// A CUDA driver call failed during upload, launch, or download.
    #[error("CUDA driver: {0}")]
    Driver(String),
    /// The driver could not allocate device memory. Transient: another
    /// process is holding the VRAM this job needs (the capacity model in
    /// [`crate::capacity`] keeps our own requests below the budget), and the
    /// memory can free again with no restart. Maps to
    /// [`quip_solver_core::SampleError::DeviceBusy`] — a single-job reject —
    /// not `DeviceFault`.
    #[error("CUDA out of memory: {0}")]
    OutOfMemory(String),
    /// The graph has more nodes than the chosen kernel's fixed-size
    /// per-thread/shared state supports. Permanent for this backend: the
    /// limit is compiled into the kernel, so retrying cannot help.
    #[error("graph N={n} exceeds self-feeding kernel limit {limit}")]
    GraphTooLarge {
        /// Node count of the rejected graph.
        n: usize,
        /// The kernel's compiled-in node ceiling.
        limit: usize,
    },
    /// The graph is well formed but this kernel cannot run it: for msa, a
    /// spin with more neighbours than the kernel's unrolled budget.
    /// Permanent for this graph and harmless to the session, so it maps to
    /// [`quip_solver_core::SampleError::Capacity`] like `GraphTooLarge`.
    #[error("graph unsupported by the kernel: {0}")]
    Unsupported(String),
    /// The persistent kernel never marked the slot COMPLETE before the
    /// driver's deadline: the device is wedged. Maps to
    /// [`quip_solver_core::SampleError::DeviceFault`] (see the `From` impl
    /// below), which ends the session for a supervisor restart rather than
    /// rejecting jobs one at a time against a device that will keep failing
    /// the same way.
    #[error("self-feeding kernel timed out")]
    KernelTimeout,
}

impl SampleError {
    /// Classify a CUDA driver result, deferring the human-readable detail.
    ///
    /// `detail` is a closure, not a `String`, because rendering a
    /// [`cudarc::driver::DriverError`] calls `cuGetErrorString` in `libcuda`.
    /// Building the message eagerly would make this classification — and
    /// every test of it — require a loaded CUDA driver, which the host-only
    /// CI runners do not have.
    fn from_driver_result(
        code: cudarc::driver::sys::CUresult,
        detail: impl FnOnce() -> String,
    ) -> Self {
        if code == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            SampleError::OutOfMemory(detail())
        } else {
            SampleError::Driver(detail())
        }
    }
}

impl From<cudarc::driver::DriverError> for SampleError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::from_driver_result(e.0, || e.to_string())
    }
}

impl From<SampleError> for quip_solver_core::SampleError {
    /// TYPE-4: name every device error variant explicitly, so a variant added
    /// to this enum later cannot silently fall through a wildcard.
    ///
    /// `GraphTooLarge` and `Unsupported` are permanent per-graph conditions
    /// (`Capacity`). `OutOfMemory` is transient VRAM pressure from another
    /// process (`DeviceBusy`: reject this job, keep the session). Every other
    /// variant here is a CUDA driver or kernel condition this backend cannot
    /// tell apart from a genuinely wedged device — cudarc surfaces a raw
    /// driver failure as an opaque string, with nothing to say whether the
    /// context can still be trusted — so each ends the session for a
    /// supervisor restart (`DeviceFault`) rather than risking a job rejected
    /// forever against hardware that will never recover.
    fn from(e: SampleError) -> Self {
        match e {
            SampleError::GraphTooLarge { .. } | SampleError::Unsupported(_) => Self::Capacity,
            SampleError::OutOfMemory(_) => Self::DeviceBusy,
            SampleError::KernelTimeout | SampleError::Cuda(_) | SampleError::Driver(_) => {
                Self::DeviceFault(e.to_string())
            }
        }
    }
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem.
///
/// # Errors
///
/// - [`SampleError::GraphTooLarge`] if `graph` has more nodes than the chosen
///   kernel's fixed-size state supports. Permanent — retrying cannot help.
/// - [`SampleError::Unsupported`] if the msa kernel cannot run this topology
///   (a spin with more than 20 neighbours). Permanent.
/// - [`SampleError::KernelTimeout`] if the persistent kernel never marks the
///   slot complete before the driver's deadline.
/// - [`SampleError::Cuda`] if opening the device or compiling its kernels failed.
/// - [`SampleError::OutOfMemory`] if the driver could not allocate device
///   memory — transient VRAM pressure, rejected per-job.
/// - [`SampleError::Driver`] if any other CUDA driver call failed while
///   allocating the session buffers, uploading the problem, launching, or
///   downloading results.
pub fn sample_ising(
    device: &CudaDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: KernelKind,
) -> Result<Vec<SamplerResult>, SampleError> {
    streaming::sample_one(device, graph, params, kernel)
}

#[cfg(test)]
mod tests {
    use super::SampleError;
    use cudarc::driver::sys::CUresult;

    /// VRAM pressure from another process must reject one job, not end the
    /// session: `DeviceFault` here would put the miner in a restart loop for
    /// a condition that clears on its own.
    #[test]
    fn oom_driver_result_maps_to_device_busy() {
        let e = SampleError::from_driver_result(CUresult::CUDA_ERROR_OUT_OF_MEMORY, || {
            "out of memory".to_owned()
        });
        match e {
            SampleError::OutOfMemory(_) => {}
            other => panic!("expected OutOfMemory, got {other:?}"),
        }
        assert_eq!(
            quip_solver_core::SampleError::from(e),
            quip_solver_core::SampleError::DeviceBusy
        );
    }

    /// Any other driver failure still ends the session for a supervisor
    /// restart — the context cannot be trusted.
    #[test]
    fn non_oom_driver_result_stays_a_device_fault() {
        let e = SampleError::from_driver_result(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS, || {
            "an illegal memory access was encountered".to_owned()
        });
        match quip_solver_core::SampleError::from(e) {
            quip_solver_core::SampleError::DeviceFault(detail) => {
                assert!(detail.contains("CUDA driver"));
            }
            other => panic!("expected DeviceFault, got {other:?}"),
        }
    }

    /// The detail closure runs only for the variant that is built, so a
    /// caller never pays `cuGetErrorString` for a message it discards.
    #[test]
    fn detail_is_rendered_once_for_the_chosen_variant() {
        let mut calls = 0;
        let e = SampleError::from_driver_result(CUresult::CUDA_ERROR_INVALID_VALUE, || {
            calls += 1;
            "invalid argument".to_owned()
        });
        assert_eq!(calls, 1);
        assert!(matches!(e, SampleError::Driver(_)));
    }

    /// A topology the kernel cannot run is permanent for that graph and
    /// harmless to the session, so it is a per-job `Capacity` reject, not
    /// a `DeviceFault` that restarts the miner.
    #[test]
    fn unsupported_graph_maps_to_capacity() {
        let e = SampleError::Unsupported("msa: topology max degree 21 exceeds 20".to_owned());
        assert_eq!(
            quip_solver_core::SampleError::from(e),
            quip_solver_core::SampleError::Capacity
        );
    }
}

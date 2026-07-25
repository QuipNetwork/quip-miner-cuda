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
use crate::streaming;
use quip_miner_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};
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
    /// The persistent kernel never marked the slot COMPLETE before the
    /// driver's deadline. Transient: the device is wedged or oversubscribed.
    #[error("self-feeding kernel timed out")]
    KernelTimeout,
}

impl From<cudarc::driver::DriverError> for SampleError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        SampleError::Driver(e.to_string())
    }
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem.
///
/// # Errors
///
/// - [`SampleError::GraphTooLarge`] if `graph` has more nodes than the chosen
///   kernel's fixed-size state supports. Permanent — retrying cannot help.
/// - [`SampleError::KernelTimeout`] if the persistent kernel never marks the
///   slot complete before the driver's deadline.
/// - [`SampleError::Cuda`] if opening the device or compiling its kernels failed.
/// - [`SampleError::Driver`] if a CUDA driver call failed while allocating the
///   session buffers, uploading the problem, launching, or downloading results.
pub fn sample_ising(
    device: &CudaDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<Vec<SamplerResult>, SampleError> {
    streaming::sample_one(device, graph, params, algorithm)
}

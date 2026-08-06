//! Node-capacity policy for the self-feeding kernels.
//!
//! The kernels size their state arrays from a `QUIP_MAX_NODES` macro that
//! `cuda_device` supplies at NVRTC compile time. This module owns every
//! bound on that value so the binaries, the device, and the backend
//! identities cannot disagree.

use quip_miner_core::Algorithm;
use thiserror::Error;

/// Shipped `unpacked_state` size in `kernels/sa.cu`. Also the floor: a
/// smaller array saves nothing measurable and only narrows what runs.
pub const SA_DEFAULT_NODES: usize = 5000;

/// CUDA's hard cap on per-thread local memory. The SA stack frame lives
/// here, so it bounds SA regardless of how much device memory is free.
pub const LOCAL_MEM_BYTES_PER_THREAD: usize = 512 * 1024;

/// Threads per block the SA kernel launches, and so the multiplier on its
/// per-thread `delta_energy` workspace. Mirrors `total_threads` in
/// `streaming::build_algo_state`.
pub const SA_THREADS_PER_NONCE: usize = 256;

/// Fraction of free device memory the capacity derivation will spend, as
/// numerator over denominator.
///
/// The model below covers the two allocations that scale with node count and
/// dominate at large N. It does not model the topology, sample and energy
/// buffers, nor the driver's own rounding of the local-memory reservation, so
/// the headroom absorbs them. The driver's `CUDA_ERROR_OUT_OF_MEMORY` is the
/// real backstop; this only keeps a reasonable request from reaching it.
pub const MEMORY_HEADROOM_NUM: usize = 4;
/// See [`MEMORY_HEADROOM_NUM`].
pub const MEMORY_HEADROOM_DEN: usize = 5;

/// Device facts the capacity derivation needs, read once at open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`. Bounds Gibbs.
    pub shared_bytes_per_block: usize,
    /// Free device memory at open, from `cuMemGetInfo`. Bounds SA.
    pub free_bytes: usize,
    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
    pub sm_count: usize,
    /// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR`. The driver
    /// reserves local memory for full occupancy, so this multiplies the SA
    /// frame whether or not that many threads ever run.
    pub threads_per_sm: usize,
}

impl DeviceLimits {
    /// Free memory less the headroom the model does not account for.
    #[must_use]
    pub fn usable_bytes(&self) -> usize {
        self.free_bytes / MEMORY_HEADROOM_DEN * MEMORY_HEADROOM_NUM
    }
}

/// Bytes of per-thread stack frame the SA kernel needs for `nodes`.
///
/// `unpacked_state` is one byte per node and `packed_state` one bit, both
/// sized from `QUIP_MAX_NODES` in `kernels/sa.cu`.
#[must_use]
pub fn sa_frame_bytes(nodes: usize) -> usize {
    nodes + nodes.div_ceil(8)
}

/// Device memory the SA path consumes at `nodes`, for the two allocations
/// that scale with node count.
///
/// The local-memory backing store is the frame times full occupancy, because
/// the driver reserves for every thread the device could run. The
/// `delta_energy` workspace is one byte per node per launched thread
/// (`streaming.rs`, `total_threads * topology.n`), and SA launches one block
/// per SM.
#[must_use]
pub fn sa_working_set_bytes(nodes: usize, limits: &DeviceLimits) -> usize {
    let local = sa_frame_bytes(nodes)
        .saturating_mul(limits.sm_count)
        .saturating_mul(limits.threads_per_sm);
    let workspace = nodes
        .saturating_mul(limits.sm_count)
        .saturating_mul(SA_THREADS_PER_NONCE);
    local.saturating_add(workspace)
}

/// Largest node count whose SA working set fits this device.
///
/// Video memory is the line. A device too full for even
/// [`SA_DEFAULT_NODES`] gets a budget below it and [`resolve`] refuses,
/// naming the request and the budget. Flooring this at the default would
/// hand that case to the driver instead, and a bare
/// `CUDA_ERROR_OUT_OF_MEMORY` from inside an allocation says nothing about
/// which knob to turn.
#[must_use]
pub fn sa_budget(limits: &DeviceLimits) -> usize {
    // Bytes per node, from `sa_working_set_bytes` with the packed-state
    // rounding dropped: 9 bytes of frame per 8 nodes, plus one workspace byte
    // per launched thread.
    let per_node = limits
        .sm_count
        .saturating_mul(limits.threads_per_sm.saturating_mul(9) / 8 + SA_THREADS_PER_NONCE)
        .max(1);
    let by_memory = limits.usable_bytes() / per_node;
    // 9 frame bytes per 8 nodes, inverted against the per-thread cap.
    let by_local_cap = LOCAL_MEM_BYTES_PER_THREAD * 8 / 9;
    by_memory.min(by_local_cap)
}

/// Shipped `shared_state` size in `kernels/gibbs.cu`, and the floor.
pub const GIBBS_DEFAULT_NODES: usize = 4800;

/// Bytes of shared memory the Gibbs kernel uses outside `shared_state`:
/// `s_chunk` and `s_arrival`, one `int` each.
pub const GIBBS_FIXED_SHARED_BYTES: usize = 8;

/// Which device limit bounds an algorithm's capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetResource {
    /// Gibbs holds its spin state in shared memory, per block.
    SharedMemory,
    /// SA holds its state in per-thread local memory, which the driver backs
    /// with device memory reserved for full occupancy.
    DeviceMemory,
}

impl std::fmt::Display for BudgetResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SharedMemory => write!(f, "shared-memory"),
            Self::DeviceMemory => write!(f, "memory"),
        }
    }
}

/// Why a capacity request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CapacityError {
    /// Above what this device can hold. `resource` names which limit bound
    /// it, because the two algorithms are bounded by different ones and a
    /// message naming the wrong resource sends the reader to the wrong knob.
    #[error("requested {requested} nodes exceeds the device {resource} budget of {budget}")]
    AboveDeviceBudget {
        /// Nodes asked for.
        requested: usize,
        /// Budget derived from the device.
        budget: usize,
        /// Which device limit bound this request.
        resource: BudgetResource,
    },
}

/// Largest Gibbs `shared_state` this device can hold, in nodes. One node is
/// one `signed char`, so bytes and nodes are the same number.
#[must_use]
pub fn gibbs_budget(shared_bytes_per_block: usize) -> usize {
    shared_bytes_per_block.saturating_sub(GIBBS_FIXED_SHARED_BYTES)
}

/// Capacity used when `--max-nodes` is absent.
#[must_use]
pub fn default_nodes(algorithm: Algorithm) -> usize {
    match algorithm {
        Algorithm::Sa => SA_DEFAULT_NODES,
        Algorithm::Gibbs => GIBBS_DEFAULT_NODES,
    }
}

/// Capacity to advertise in `--capabilities`, which must answer without
/// opening the device.
///
/// Applies the floor and every bound knowable without a device. Both
/// algorithms are now bounded by device properties rather than a static
/// ceiling, so the request passes through and `open_with_nodes` is what
/// refuses. The per-thread local-memory cap is the one hardware bound that
/// holds on every CUDA device, so SA is clamped to it here.
#[must_use]
pub fn advertised_nodes(algorithm: Algorithm, requested: usize) -> usize {
    let want = requested.max(default_nodes(algorithm));
    match algorithm {
        Algorithm::Sa => want.min(LOCAL_MEM_BYTES_PER_THREAD * 8 / 9),
        Algorithm::Gibbs => want,
    }
}

/// Resolve a requested capacity against the algorithm and the device.
///
/// A request below the default is raised to the default. A request above a
/// bound is an error, never a silent clamp.
///
/// # Errors
///
/// [`CapacityError::AboveDeviceBudget`] when the request exceeds
/// [`sa_budget`] or [`gibbs_budget`] for this device.
pub fn resolve(
    algorithm: Algorithm,
    requested: usize,
    limits: &DeviceLimits,
) -> Result<usize, CapacityError> {
    let floor = default_nodes(algorithm);
    let want = requested.max(floor);
    let (budget, resource) = match algorithm {
        Algorithm::Sa => (sa_budget(limits), BudgetResource::DeviceMemory),
        Algorithm::Gibbs => (
            gibbs_budget(limits.shared_bytes_per_block),
            BudgetResource::SharedMemory,
        ),
    };
    if want > budget {
        return Err(CapacityError::AboveDeviceBudget {
            requested: want,
            budget,
            resource,
        });
    }
    Ok(want)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_miner_core::Algorithm;

    /// A4000 reports 49152 bytes of shared memory per block. `s_chunk` and
    /// `s_arrival` take 8 of them, so 49144 nodes is the Gibbs ceiling.
    /// Measured: 48000 runs, 65536 fails to compile.
    #[test]
    fn gibbs_budget_subtracts_the_fixed_shared_members() {
        assert_eq!(gibbs_budget(49152), 49144);
    }

    #[test]
    fn resolve_accepts_a_request_inside_both_bounds() {
        let limits = a4000(A4000_FREE);
        assert_eq!(resolve(Algorithm::Gibbs, 5640, &limits), Ok(5640));
        assert_eq!(resolve(Algorithm::Sa, 5640, &limits), Ok(5640));
    }

    #[test]
    fn resolve_rejects_gibbs_above_the_device_budget() {
        assert_eq!(
            resolve(Algorithm::Gibbs, 65536, &a4000(A4000_FREE)),
            Err(CapacityError::AboveDeviceBudget {
                requested: 65536,
                budget: 49144,
                resource: BudgetResource::SharedMemory,
            })
        );
    }

    /// A request below the default would shrink the kernel array for no
    /// gain, so the default is a floor.
    #[test]
    fn resolve_raises_a_small_request_to_the_default() {
        let limits = a4000(A4000_FREE);
        assert_eq!(resolve(Algorithm::Sa, 64, &limits), Ok(SA_DEFAULT_NODES));
        assert_eq!(
            resolve(Algorithm::Gibbs, 64, &limits),
            Ok(GIBBS_DEFAULT_NODES)
        );
    }

    #[test]
    fn defaults_match_the_shipped_kernel_arrays() {
        assert_eq!(default_nodes(Algorithm::Sa), 5000);
        assert_eq!(default_nodes(Algorithm::Gibbs), 4800);
    }

    /// An A4000 as this code sees it: 48 SMs, 1536 resident threads each,
    /// 48152 bytes of shared memory per block, and a nominally free 16 GiB.
    const A4000_SMS: usize = 48;
    const A4000_THREADS_PER_SM: usize = 1536;
    const A4000_FREE: usize = 16 * 1024 * 1024 * 1024;

    fn a4000(free_bytes: usize) -> DeviceLimits {
        DeviceLimits {
            shared_bytes_per_block: 49152,
            free_bytes,
            sm_count: A4000_SMS,
            threads_per_sm: A4000_THREADS_PER_SM,
        }
    }

    /// The SA stack frame is `unpacked_state` (one byte per node) plus
    /// `packed_state` (one bit per node), both sized from `QUIP_MAX_NODES`.
    #[test]
    fn sa_frame_is_the_two_state_arrays() {
        assert_eq!(sa_frame_bytes(5000), 5000 + 625);
        assert_eq!(sa_frame_bytes(8), 8 + 1);
        // Rounds up: 9 nodes need 2 packed bytes, not 1.
        assert_eq!(sa_frame_bytes(9), 9 + 2);
    }

    /// The budget must be self-consistent: the working set at the budget fits
    /// in the free memory it was derived from, and one node more does not.
    #[test]
    fn sa_budget_is_the_largest_node_count_that_fits() {
        let limits = a4000(A4000_FREE);
        let budget = sa_budget(&limits);
        assert!(budget > 0, "a 16 GiB card must afford some capacity");
        assert!(
            sa_working_set_bytes(budget, &limits) <= limits.usable_bytes(),
            "the working set at the budget must fit"
        );
        assert!(
            sa_working_set_bytes(budget + 1, &limits) > limits.usable_bytes(),
            "one node above the budget must not fit"
        );
    }

    /// A bigger card affords more, which is the whole reason for deriving
    /// this rather than hardcoding it.
    #[test]
    fn sa_budget_scales_with_free_memory() {
        let small = sa_budget(&a4000(8 * 1024 * 1024 * 1024));
        let large = sa_budget(&a4000(48 * 1024 * 1024 * 1024));
        assert!(
            large > small,
            "48 GiB must afford more than 8 GiB: {large} vs {small}"
        );
    }

    /// CUDA caps local memory at 512 KiB per thread regardless of how much
    /// device memory is free, so a huge card is still bounded.
    #[test]
    fn sa_budget_respects_the_per_thread_local_cap() {
        let huge = sa_budget(&a4000(1024 * 1024 * 1024 * 1024));
        assert!(
            sa_frame_bytes(huge) <= LOCAL_MEM_BYTES_PER_THREAD,
            "frame {} exceeds the {LOCAL_MEM_BYTES_PER_THREAD}-byte per-thread cap",
            sa_frame_bytes(huge)
        );
    }

    /// At the limit resolves, one above it fails. This is the contract the
    /// whole module exists for.
    #[test]
    fn resolve_sa_accepts_at_the_budget_and_rejects_above_it() {
        let limits = a4000(A4000_FREE);
        let budget = sa_budget(&limits);
        assert_eq!(resolve(Algorithm::Sa, budget, &limits), Ok(budget));
        assert_eq!(
            resolve(Algorithm::Sa, budget + 1, &limits),
            Err(CapacityError::AboveDeviceBudget {
                requested: budget + 1,
                budget,
                resource: BudgetResource::DeviceMemory,
            })
        );
    }

    /// Same contract for Gibbs, whose budget comes from shared memory.
    #[test]
    fn resolve_gibbs_accepts_at_the_budget_and_rejects_above_it() {
        let limits = a4000(A4000_FREE);
        let budget = gibbs_budget(limits.shared_bytes_per_block);
        assert_eq!(budget, 49144);
        assert_eq!(resolve(Algorithm::Gibbs, budget, &limits), Ok(budget));
        assert_eq!(
            resolve(Algorithm::Gibbs, budget + 1, &limits),
            Err(CapacityError::AboveDeviceBudget {
                requested: budget + 1,
                budget,
                resource: BudgetResource::SharedMemory,
            })
        );
    }

    /// Video memory is the line. A card too full for even the default must
    /// report a budget below it and refuse with our message, rather than be
    /// floored to the default and hand a bare out-of-memory to the driver.
    #[test]
    fn a_starved_card_is_refused_rather_than_left_to_the_driver() {
        let limits = a4000(1024 * 1024);
        let budget = sa_budget(&limits);
        assert!(
            budget < SA_DEFAULT_NODES,
            "a 1 MiB card cannot afford the default: budget {budget}"
        );
        assert_eq!(
            resolve(Algorithm::Sa, SA_DEFAULT_NODES, &limits),
            Err(CapacityError::AboveDeviceBudget {
                requested: SA_DEFAULT_NODES,
                budget,
                resource: BudgetResource::DeviceMemory,
            })
        );
    }

    /// A card with room for the default still opens on a plain invocation.
    #[test]
    fn an_ordinary_card_affords_the_default() {
        let limits = a4000(A4000_FREE);
        assert_eq!(
            resolve(Algorithm::Sa, SA_DEFAULT_NODES, &limits),
            Ok(SA_DEFAULT_NODES)
        );
    }

    /// `--capabilities` runs without opening the device, so it cannot call
    /// `resolve`. The per-thread local-memory cap is the one bound that holds
    /// on every CUDA device, so it is the only clamp available here.
    #[test]
    fn advertised_clamps_sa_to_the_per_thread_local_cap() {
        let cap = LOCAL_MEM_BYTES_PER_THREAD * 8 / 9;
        assert_eq!(advertised_nodes(Algorithm::Sa, cap * 2), cap);
        assert_eq!(advertised_nodes(Algorithm::Sa, 5640), 5640);
        assert_eq!(advertised_nodes(Algorithm::Sa, 64), SA_DEFAULT_NODES);
    }

    /// Gibbs has no static ceiling — its bound comes from the device — so the
    /// request passes through and `open_with_nodes` is what refuses.
    #[test]
    fn advertised_passes_gibbs_through_above_the_default() {
        assert_eq!(advertised_nodes(Algorithm::Gibbs, 32768), 32768);
        assert_eq!(advertised_nodes(Algorithm::Gibbs, 64), GIBBS_DEFAULT_NODES);
    }
}

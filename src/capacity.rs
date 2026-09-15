//! Node-capacity policy for the self-feeding kernels.
//!
//! The kernels size their state arrays from a `QUIP_MAX_NODES` macro that
//! `cuda_device` supplies at NVRTC compile time. This module owns every
//! bound on that value so the binaries, the device, and the backend
//! identities cannot disagree.

use crate::kernel::KernelKind;
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
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`. Bounds msa,
    /// whose block opts in to more than the default per-block limit.
    pub shared_bytes_per_block_optin: usize,
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

/// Capacity used when `--max-nodes` is absent for `quip-cuda-msa`. The same
/// floor as SA so the three binaries default alike. The msa kernel has no
/// compiled-in array: its spin state is dynamic shared memory sized per
/// session, so this floor only says what a device must hold to open.
pub const MSA_DEFAULT_NODES: usize = 5000;

/// Reads packed into one 64-bit word by the msa kernel (`kernels/msa.cu`).
pub const MSA_LANES: usize = 64;

/// Replica words per spin the msa session allocates for. Two words is 128
/// reads, which is what the adapt envelope pins and what a 99 KB opt-in
/// holds at Zephyr scale. Four words would need 146 KB for 4577 spins.
pub const MSA_REPLICA_WORDS: usize = 2;

/// Read cap for the msa kernel: `MSA_LANES * MSA_REPLICA_WORDS`.
pub const MSA_MAX_READS: usize = MSA_LANES * MSA_REPLICA_WORDS;

/// Threads per block the msa kernel launches. Measured faster than 512 and
/// 1024 on an RTX 5090 (MR !26). Mirrors `block_dim` in `streaming::launch`.
pub const MSA_THREADS_PER_NONCE: usize = 256;

/// Neighbours per spin the msa kernel unrolls (`MSA_MAX_DEG` in
/// `kernels/msa.cu`). Zephyr's degree is 20. A denser topology is refused
/// per job.
pub const MSA_MAX_DEGREE: usize = 20;

/// Static `__shared__` members of the msa kernel: `s_active_slot` and
/// `s_abort`, one `int` each (8 bytes of variables). The compiled function
/// reports 16: measured opening `kernels/msa.cu` under NVRTC 12.9 on an
/// A4000 (`sm_86`), where the static shared allocation rounds up to a
/// 16-byte boundary. They count against the opt-in cap, so the dynamic
/// budget is the opt-in less this. `CudaDevice::open_with_nodes` checks the
/// loaded function reports exactly this, so the constant cannot drift from
/// the kernel.
pub const MSA_STATIC_SHARED_BYTES: usize = 16;

/// Dynamic shared memory the msa kernel uses outside the spin state: the
/// 8192-byte threshold row plus 64 `u64` acceptance cuts (`MSA_ROW` and
/// `MSA_MAX_FIELD + 1` in `kernels/msa.cu`).
pub const MSA_FIXED_SHARED_BYTES: usize = 8192 + 64 * 8;

/// Dynamic shared memory one msa block may ask for on a device with this
/// opt-in ceiling: the ceiling less the kernel's static members.
#[must_use]
pub fn msa_dynamic_shared_bytes(shared_bytes_per_block_optin: usize) -> usize {
    shared_bytes_per_block_optin.saturating_sub(MSA_STATIC_SHARED_BYTES)
}

/// Bytes of dynamic shared memory an msa session needs for `nodes` spins at
/// `words` replica words each.
#[must_use]
pub fn msa_shared_bytes(nodes: usize, words: usize) -> usize {
    nodes
        .saturating_mul(words)
        .saturating_mul(8)
        .saturating_add(MSA_FIXED_SHARED_BYTES)
}

/// Largest node count whose msa state fits `dynamic_shared_bytes` at
/// [`MSA_REPLICA_WORDS`], the shape every job up to [`MSA_MAX_READS`] reads
/// uses.
#[must_use]
pub fn msa_budget(dynamic_shared_bytes: usize) -> usize {
    dynamic_shared_bytes.saturating_sub(MSA_FIXED_SHARED_BYTES) / (MSA_REPLICA_WORDS * 8)
}

/// Which device limit bounds a kernel's capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetResource {
    /// Gibbs and msa hold their spin state in shared memory, per block.
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
    /// it, because SA is bounded by device memory while msa and Gibbs are
    /// bounded by shared memory, and a message naming the wrong resource
    /// sends the reader to the wrong knob.
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
pub fn default_nodes(kernel: KernelKind) -> usize {
    match kernel {
        KernelKind::Sa => SA_DEFAULT_NODES,
        KernelKind::Msa => MSA_DEFAULT_NODES,
        KernelKind::Gibbs => GIBBS_DEFAULT_NODES,
    }
}

/// Capacity to advertise in `--capabilities`, which must answer without
/// opening the device.
///
/// Applies the floor and every bound knowable without a device. Every
/// kernel is now bounded by device properties rather than a static
/// ceiling, so the request passes through and `open_with_nodes` is what
/// refuses. SA is clamped to the per-thread local-memory cap here; msa and
/// Gibbs are bounded by device shared memory, so their requests pass
/// through.
#[must_use]
pub fn advertised_nodes(kernel: KernelKind, requested: usize) -> usize {
    let want = requested.max(default_nodes(kernel));
    match kernel {
        KernelKind::Sa => want.min(LOCAL_MEM_BYTES_PER_THREAD * 8 / 9),
        KernelKind::Msa | KernelKind::Gibbs => want,
    }
}

/// Resolve a requested capacity against the kernel and the device.
///
/// A request below the default is raised to the default. A request above a
/// bound is an error, never a silent clamp.
///
/// # Errors
///
/// [`CapacityError::AboveDeviceBudget`] when the request exceeds
/// [`sa_budget`], [`msa_budget`] or [`gibbs_budget`] for this device.
pub fn resolve(
    kernel: KernelKind,
    requested: usize,
    limits: &DeviceLimits,
) -> Result<usize, CapacityError> {
    let floor = default_nodes(kernel);
    let want = requested.max(floor);
    let (budget, resource) = match kernel {
        KernelKind::Sa => (sa_budget(limits), BudgetResource::DeviceMemory),
        KernelKind::Msa => (
            msa_budget(msa_dynamic_shared_bytes(
                limits.shared_bytes_per_block_optin,
            )),
            BudgetResource::SharedMemory,
        ),
        KernelKind::Gibbs => (
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
    use crate::kernel::KernelKind;

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
        assert_eq!(resolve(KernelKind::Gibbs, 5640, &limits), Ok(5640));
        assert_eq!(resolve(KernelKind::Sa, 5640, &limits), Ok(5640));
    }

    #[test]
    fn resolve_rejects_gibbs_above_the_device_budget() {
        assert_eq!(
            resolve(KernelKind::Gibbs, 65536, &a4000(A4000_FREE)),
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
        assert_eq!(resolve(KernelKind::Sa, 64, &limits), Ok(SA_DEFAULT_NODES));
        assert_eq!(resolve(KernelKind::Msa, 64, &limits), Ok(MSA_DEFAULT_NODES));
        assert_eq!(
            resolve(KernelKind::Gibbs, 64, &limits),
            Ok(GIBBS_DEFAULT_NODES)
        );
    }

    #[test]
    fn defaults_match_the_shipped_kernel_arrays() {
        assert_eq!(default_nodes(KernelKind::Sa), 5000);
        assert_eq!(default_nodes(KernelKind::Msa), 5000);
        assert_eq!(default_nodes(KernelKind::Gibbs), 4800);
    }

    /// An A4000 as this code sees it: 48 SMs, 1536 resident threads each,
    /// 48152 bytes of shared memory per block, and a nominally free 16 GiB.
    const A4000_SMS: usize = 48;
    const A4000_THREADS_PER_SM: usize = 1536;
    const A4000_FREE: usize = 16 * 1024 * 1024 * 1024;

    fn a4000(free_bytes: usize) -> DeviceLimits {
        DeviceLimits {
            shared_bytes_per_block: 49152,
            shared_bytes_per_block_optin: 101_376,
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
        assert_eq!(resolve(KernelKind::Sa, budget, &limits), Ok(budget));
        assert_eq!(
            resolve(KernelKind::Sa, budget + 1, &limits),
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
        assert_eq!(resolve(KernelKind::Gibbs, budget, &limits), Ok(budget));
        assert_eq!(
            resolve(KernelKind::Gibbs, budget + 1, &limits),
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
            resolve(KernelKind::Sa, SA_DEFAULT_NODES, &limits),
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
            resolve(KernelKind::Sa, SA_DEFAULT_NODES, &limits),
            Ok(SA_DEFAULT_NODES)
        );
    }

    /// `--capabilities` runs without opening the device, so it cannot call
    /// `resolve`. The per-thread local-memory cap is the one bound that holds
    /// on every CUDA device, so it is the only clamp available here.
    #[test]
    fn advertised_clamps_sa_to_the_per_thread_local_cap() {
        let cap = LOCAL_MEM_BYTES_PER_THREAD * 8 / 9;
        assert_eq!(advertised_nodes(KernelKind::Sa, cap * 2), cap);
        assert_eq!(advertised_nodes(KernelKind::Sa, 5640), 5640);
        assert_eq!(advertised_nodes(KernelKind::Sa, 64), SA_DEFAULT_NODES);
    }

    /// Gibbs has no static ceiling — its bound comes from the device — so the
    /// request passes through and `open_with_nodes` is what refuses.
    #[test]
    fn advertised_passes_gibbs_through_above_the_default() {
        assert_eq!(advertised_nodes(KernelKind::Gibbs, 32768), 32768);
        assert_eq!(advertised_nodes(KernelKind::Gibbs, 64), GIBBS_DEFAULT_NODES);
    }

    /// An A4000 opts in to 101376 bytes (99 KB) per block. Less the two
    /// static ints and the 8704-byte threshold structures, two 8-byte
    /// words per spin fit 5791 spins.
    #[test]
    fn msa_budget_on_a_99_kb_optin() {
        assert_eq!(msa_dynamic_shared_bytes(101_376), 101_360);
        assert_eq!(msa_budget(101_360), 5791);
        assert!(msa_shared_bytes(5791, MSA_REPLICA_WORDS) <= 101_360);
        assert!(msa_shared_bytes(5792, MSA_REPLICA_WORDS) > 101_360);
    }

    /// The other opt-in sizes in the supported-arch table. Volta (96 KB),
    /// Ampere datacenter (163 KB) and Hopper (227 KB) hold the default.
    /// Turing (64 KB) does not.
    #[test]
    fn msa_budget_across_the_supported_optin_sizes() {
        assert_eq!(msa_budget(msa_dynamic_shared_bytes(98_304)), 5599);
        assert_eq!(msa_budget(msa_dynamic_shared_bytes(166_912)), 9887);
        assert_eq!(msa_budget(msa_dynamic_shared_bytes(232_448)), 13_983);
        assert_eq!(msa_budget(msa_dynamic_shared_bytes(65_536)), 3551);
    }

    #[test]
    fn resolve_msa_accepts_at_the_budget_and_rejects_above_it() {
        let limits = a4000(A4000_FREE);
        assert_eq!(resolve(KernelKind::Msa, 5791, &limits), Ok(5791));
        assert_eq!(
            resolve(KernelKind::Msa, 5792, &limits),
            Err(CapacityError::AboveDeviceBudget {
                requested: 5792,
                budget: 5791,
                resource: BudgetResource::SharedMemory,
            })
        );
    }

    /// Turing cannot hold the default at 128 reads, so the open refuses
    /// with the shared-memory budget instead of rejecting every job later.
    #[test]
    fn resolve_refuses_msa_on_a_turing_sized_optin() {
        let mut limits = a4000(A4000_FREE);
        limits.shared_bytes_per_block_optin = 65_536;
        assert_eq!(
            resolve(KernelKind::Msa, MSA_DEFAULT_NODES, &limits),
            Err(CapacityError::AboveDeviceBudget {
                requested: MSA_DEFAULT_NODES,
                budget: 3551,
                resource: BudgetResource::SharedMemory,
            })
        );
    }

    #[test]
    fn msa_read_cap_is_two_words_of_lanes() {
        assert_eq!(MSA_MAX_READS, 128);
    }

    /// msa has no static ceiling either; the device budget is what refuses.
    #[test]
    fn advertised_passes_msa_through_above_the_default() {
        assert_eq!(advertised_nodes(KernelKind::Msa, 8192), 8192);
        assert_eq!(advertised_nodes(KernelKind::Msa, 64), MSA_DEFAULT_NODES);
    }
}

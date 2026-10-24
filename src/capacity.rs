//! Node-capacity policy for the self-feeding kernels.
//!
//! The sa and gibbs kernels size their state arrays from a `QUIP_MAX_NODES`
//! macro that `cuda_device` supplies at NVRTC compile time. The msa kernel
//! keeps its hot spin state in dynamic shared memory sized per launch and
//! spills the rest to a global buffer. This module
//! owns every bound on those values so the binaries, the device, and the
//! backend identities cannot disagree.

use crate::coefficients::{CoefficientPath, CoefficientPlan};
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

/// Bytes per node per thread of the wide SA `delta_energy` workspace (`int`).
pub const SA_WIDE_DELTA_BYTES: usize = 4;

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

/// Concurrent nonce groups at the kernel's fixed SM width.
#[must_use]
pub fn instances(kernel: KernelKind, sm_count: usize) -> usize {
    (sm_count / kernel.sms_per_instance()).max(1)
}

/// Coefficient storage and arithmetic for both fields and couplings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Unit couplings and fields in {-1, 0, 1}, stored as i8.
    Binary,
    /// Scaled i8 coefficients with i32 arithmetic.
    Int8,
    /// Exact coefficients wider than i8, stored as i32.
    Int32,
    /// Real coefficients stored and evaluated as f32.
    Float32,
}

impl Precision {
    /// Map the selected coefficient path and unit-model property to precision.
    #[must_use]
    pub fn from_plan(plan: CoefficientPlan, unit: bool) -> Self {
        match plan.path {
            CoefficientPath::Int8 if unit => Self::Binary,
            CoefficientPath::Int8 => Self::Int8,
            CoefficientPath::Wide => Self::Int32,
        }
    }
}

/// Device representation of the MSA coupling matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsaStorage {
    /// Two directed CSR entries per undirected coupling.
    Sparse,
    /// Dense matrix storage, not yet implemented.
    Dense,
}

/// Topology sizes needed to bound session allocations without building CSR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// Number of spins and fields.
    pub nodes: usize,
    /// Number of undirected couplings.
    pub couplers: usize,
    /// Largest node degree, bounding the greedy colouring by degree plus one.
    pub max_degree: usize,
}

/// Device memory shared by the session and repeated for each nonce group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    /// Session-wide device allocations, including topology and lane tables.
    /// This is global memory, distinct from CUDA block shared memory.
    pub shared_bytes: usize,
    /// Device memory per concurrent nonce group.
    pub per_instance_bytes: usize,
    /// Concurrent nonce groups, derived from the device's SM count.
    pub instances: usize,
}

impl Footprint {
    /// Total device memory, saturating so oversized shapes cannot wrap to fit.
    #[must_use]
    pub fn total(&self) -> usize {
        self.shared_bytes
            .saturating_add(self.per_instance_bytes.saturating_mul(self.instances))
    }
}

/// Upper estimate of sparse MSA session device allocations at full width.
///
/// CSR and colour tables, beta schedules and integer acceptance tables are
/// session-wide. Each instance owns three coefficient/sample/energy slots,
/// eight control words and its spin spill. Empty buffers get stand-ins like
/// `SelfFeedingSession::build`. Colour counts use the greedy degree-plus-one
/// bound. A 32 MiB allowance covers CUDA pool growth and allocation overhead
/// measured by the A4000 allocation test.
///
/// Schedule tables use `betas.max(1)` entries, matching the session builders.
/// This assumes the default shared-spin layout, without a bench shared-spin
/// cap. Dense storage returns a saturated per-instance size until its
/// allocation layout is implemented.
#[must_use]
pub fn msa_footprint(
    shape: ModelShape,
    precision: Precision,
    storage: MsaStorage,
    reads: usize,
    betas: usize,
    limits: &DeviceLimits,
) -> Footprint {
    let count = instances(KernelKind::Msa, limits.sm_count);
    if storage == MsaStorage::Dense {
        return Footprint {
            shared_bytes: 0,
            per_instance_bytes: usize::MAX,
            instances: count,
        };
    }
    let betas = betas.max(1);
    let n = shape.nodes;
    let nnz = shape.couplers.saturating_mul(2).max(1);
    let colours = n.min(shape.max_degree.saturating_add(1)).max(1);
    let mut shared = n
        .saturating_add(1) // d_row_ptr
        .saturating_add(nnz) // d_col_ind
        .saturating_add(colours.saturating_mul(2)) // starts and counts
        .saturating_add(n.max(1)) // colour nodes
        .saturating_mul(4)
        .saturating_add(betas.saturating_mul(4)) // d_beta
        .saturating_add(32 << 20); // CUDA pool growth and allocation overhead
    let coefficient_bytes = match precision {
        Precision::Binary | Precision::Int8 => 1,
        Precision::Int32 | Precision::Float32 => 4,
    };
    match precision {
        Precision::Int8 | Precision::Int32 => {
            shared = shared
                .saturating_add(betas.saturating_mul(8 + 4)) // beta_fixed and delta_cut
                .saturating_add(crate::accept_table::NEGLOG_LEN * 8);
        }
        Precision::Binary | Precision::Float32 => {}
    }
    let reads = reads.max(1);
    let words = reads
        .div_ceil(MSA_LANES)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    let shared_spins = n.min(msa_shared_spins(
        msa_dynamic_shared_bytes(limits.shared_bytes_per_block_optin),
        words,
    ));
    let spill = msa_spill_words(n, shared_spins, words, 1)
        .max(1)
        .saturating_mul(8);
    let per_instance = n
        .max(1)
        .saturating_add(nnz)
        .saturating_mul(coefficient_bytes)
        .saturating_mul(3) // d_h and d_j, three slots
        .saturating_add(reads.saturating_mul(n.div_ceil(8).max(1)).saturating_mul(3))
        .saturating_add(reads.saturating_mul(4).saturating_mul(3)) // energies
        .saturating_add(8 * 4) // d_ctrl
        .saturating_add(spill);
    Footprint {
        shared_bytes: shared,
        per_instance_bytes: per_instance,
        instances: count,
    }
}

/// Usable device memory per MSA instance, before session-wide allocations.
#[must_use]
pub fn msa_per_instance_budget(limits: &DeviceLimits) -> usize {
    limits.usable_bytes() / instances(KernelKind::Msa, limits.sm_count)
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

/// Device memory the wide SA path consumes at `nodes`: the local frame compiled
/// at `frame_nodes`, an i32 delta workspace, and the `-ln(u)` table.
#[must_use]
pub fn sa_wide_working_set_bytes(nodes: usize, frame_nodes: usize, limits: &DeviceLimits) -> usize {
    let local = sa_frame_bytes(frame_nodes)
        .saturating_mul(limits.sm_count)
        .saturating_mul(limits.threads_per_sm);
    let workspace = nodes
        .saturating_mul(SA_WIDE_DELTA_BYTES)
        .saturating_mul(limits.sm_count)
        .saturating_mul(SA_THREADS_PER_NONCE);
    let table = crate::accept_table::NEGLOG_LEN * std::mem::size_of::<i64>();
    local.saturating_add(workspace).saturating_add(table)
}

/// Largest node count whose wide SA working set fits this device with the
/// compiled `frame_nodes` frame, at most `frame_nodes`. Returns zero if none fit.
#[must_use]
pub fn sa_wide_budget(frame_nodes: usize, limits: &DeviceLimits) -> usize {
    let fixed = sa_wide_working_set_bytes(0, frame_nodes, limits);
    let per_node = limits
        .sm_count
        .saturating_mul(SA_THREADS_PER_NONCE)
        .saturating_mul(SA_WIDE_DELTA_BYTES);
    let usable = limits.usable_bytes();
    let by_memory = usable
        .saturating_sub(fixed)
        .checked_div(per_node)
        .unwrap_or(frame_nodes);
    let mut nodes = by_memory.min(frame_nodes);
    while nodes > 0 && sa_wide_working_set_bytes(nodes, frame_nodes, limits) > usable {
        nodes -= 1;
    }
    nodes
}

/// Shipped `shared_state` size in `kernels/gibbs.cu`, and the floor.
pub const GIBBS_DEFAULT_NODES: usize = 4800;

/// Bytes of shared memory the Gibbs kernel uses outside `shared_state`:
/// `s_chunk` and `s_arrival`, one `int` each.
pub const GIBBS_FIXED_SHARED_BYTES: usize = 8;

/// Capacity used when `--max-nodes` is absent for `quip-cuda-msa`. The same
/// floor as SA so the three binaries default alike. The msa kernel has no
/// compiled-in array: its spin state is dynamic shared memory with a global
/// spill buffer, so this floor only sets what `--max-nodes` defaults to.
pub const MSA_DEFAULT_NODES: usize = 5000;

/// Most nodes `quip-cuda-msa` accepts. Spins past the shared-memory budget
/// spill to global memory, so this is a policy ceiling, not a device one:
/// the size the spill path is measured at (docs/perf/2026-10-msa-dram-spill-a4000.md).
pub const MSA_MAX_NODES: usize = 200_000;

/// Most edges `quip-cuda-msa` accepts: 200,000 nodes at an average degree of 40.
pub const MSA_MAX_EDGES: usize = 4_000_000;

/// Reads packed into one 64-bit word by the msa kernel (`kernels/msa.cu`).
pub const MSA_LANES: usize = 64;

/// Replica words per spin an msa session allocates at 128 reads. Shared memory
/// no longer bounds this: spins that do not fit spill to global memory.
pub const MSA_REPLICA_WORDS: usize = 2;

/// Read cap for the msa kernel at [`MSA_REPLICA_WORDS`].
pub const MSA_MAX_READS: usize = MSA_LANES * MSA_REPLICA_WORDS;

/// Threads per block the msa kernel launches. Measured faster than 512 and
/// 1024 on an RTX 5090 (MR !26). `streaming::algo_limits` carries it to the
/// launch as the block size.
pub const MSA_THREADS_PER_NONCE: u32 = 256;

/// Neighbours per spin the msa kernel counts (`MSA_MAX_DEG` in
/// `kernels/msa.cu`), 20 per pass. With the field bond the count reaches 63,
/// the most the kernel's six bit planes hold, and the same ceiling as
/// `quip-cpu-msa`. A denser topology is refused per job.
pub const MSA_MAX_DEGREE: usize = 62;

/// Static `__shared__` members of the msa kernel: `s_active_slot` and
/// `s_abort`, one `int` each (8 bytes of variables). The compiled function
/// reports 16: measured opening `kernels/msa.cu` on an A4000 (`sm_86`,
/// driver 610.57), where the driver rounds the static shared allocation
/// up to a 16-byte boundary. They count against the opt-in cap, so the dynamic
/// budget is the opt-in less this. `CudaDevice::open_with_nodes` refuses a
/// loaded function that reports more, so the budget never over-advertises.
/// A function that reports less leaves the budget conservative by under
/// one spin.
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

/// Most spins whose msa state fits `dynamic_shared_bytes` at `words` replica
/// words per spin. Spins past this spill to global memory.
#[must_use]
pub fn msa_shared_spins(dynamic_shared_bytes: usize, words: usize) -> usize {
    let per_node = words.saturating_mul(8).max(1);
    dynamic_shared_bytes.saturating_sub(MSA_FIXED_SHARED_BYTES) / per_node
}

/// `u64` words of global spill buffer an msa session allocates: the spins
/// past `shared_spins`, at `words` replica words each, for every nonce.
#[must_use]
pub fn msa_spill_words(nodes: usize, shared_spins: usize, words: usize, nonces: usize) -> usize {
    nodes
        .saturating_sub(shared_spins)
        .saturating_mul(words)
        .saturating_mul(nonces)
}

/// Largest node count whose worst-case msa spill fits this device's usable
/// memory: every spin spilled, at [`MSA_REPLICA_WORDS`] words, for each of the
/// `sm_count` nonces (msa runs one nonce per SM).
#[must_use]
pub fn msa_spill_budget(limits: &DeviceLimits) -> usize {
    let per_node = MSA_REPLICA_WORDS
        .saturating_mul(8)
        .saturating_mul(limits.sm_count)
        .max(1);
    limits.usable_bytes() / per_node
}

/// Which device limit bounds a kernel's capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetResource {
    /// Gibbs holds its spin state in shared memory, per block.
    SharedMemory,
    /// SA holds its state in per-thread local memory, which the driver backs
    /// with device memory reserved for full occupancy. msa spills spins past
    /// its shared-memory budget to a per-nonce global buffer, so its worst
    /// case is bounded by device memory too.
    DeviceMemory,
    /// A policy ceiling of the kernel, independent of the device.
    KernelCeiling,
}

impl std::fmt::Display for BudgetResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SharedMemory => write!(f, "shared-memory"),
            Self::DeviceMemory => write!(f, "memory"),
            Self::KernelCeiling => write!(f, "kernel-ceiling"),
        }
    }
}

/// Why a capacity request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CapacityError {
    /// Above what this device can hold. `resource` names which limit bound
    /// it, because SA is bounded by device memory, Gibbs by shared memory and
    /// msa by its own ceiling or by the memory its spill needs, and a message
    /// naming the wrong resource sends the reader to the wrong knob.
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
/// Applies the floor and every bound knowable without a device. SA is
/// clamped to the per-thread local-memory cap here and msa to
/// [`MSA_MAX_NODES`]. Gibbs is bounded by device shared memory, so its
/// request passes through and `open_with_nodes` is what refuses.
#[must_use]
pub fn advertised_nodes(kernel: KernelKind, requested: usize) -> usize {
    let want = requested.max(default_nodes(kernel));
    match kernel {
        KernelKind::Sa => want.min(LOCAL_MEM_BYTES_PER_THREAD * 8 / 9),
        KernelKind::Msa => want.min(MSA_MAX_NODES),
        KernelKind::Gibbs => want,
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
/// [`sa_budget`] or [`gibbs_budget`] for this device, [`MSA_MAX_NODES`]
/// (`KernelCeiling`), or for msa [`msa_spill_budget`] (`DeviceMemory`), the
/// nodes whose worst-case spill fits device memory. The kernel ceiling is
/// checked first.
pub fn resolve(
    kernel: KernelKind,
    requested: usize,
    limits: &DeviceLimits,
) -> Result<usize, CapacityError> {
    let floor = default_nodes(kernel);
    let want = requested.max(floor);
    let (budget, resource) = match kernel {
        KernelKind::Sa => (sa_budget(limits), BudgetResource::DeviceMemory),
        KernelKind::Msa if want > MSA_MAX_NODES => (MSA_MAX_NODES, BudgetResource::KernelCeiling),
        KernelKind::Msa => (msa_spill_budget(limits), BudgetResource::DeviceMemory),
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
    use crate::coefficients::{CoefficientPath, CoefficientPlan};
    use crate::kernel::KernelKind;

    fn a4000_limits() -> DeviceLimits {
        DeviceLimits {
            shared_bytes_per_block: 49_152,
            shared_bytes_per_block_optin: 101_376,
            free_bytes: 15 << 30,
            sm_count: 48,
            threads_per_sm: 1536,
        }
    }

    #[test]
    fn instances_divide_sms_by_kernel_width() {
        assert_eq!(instances(KernelKind::Msa, 48), 48);
        assert_eq!(instances(KernelKind::Sa, 48), 48);
        assert_eq!(instances(KernelKind::Gibbs, 48), 12);
        assert_eq!(instances(KernelKind::Gibbs, 3), 1);
        assert_eq!(instances(KernelKind::Msa, 0), 1);
    }

    #[test]
    fn footprint_grows_with_couplers_and_lane_width() {
        let l = a4000_limits();
        let s = |c| ModelShape {
            nodes: 100_000,
            couplers: c,
            max_degree: 40,
        };
        let u = msa_footprint(
            s(2_000_000),
            Precision::Binary,
            MsaStorage::Sparse,
            128,
            1,
            &l,
        );
        let w = msa_footprint(
            s(2_000_000),
            Precision::Int32,
            MsaStorage::Sparse,
            128,
            1,
            &l,
        );
        assert!(w.total() > u.total());
        assert!(
            msa_footprint(
                s(4_000_000),
                Precision::Binary,
                MsaStorage::Sparse,
                128,
                1,
                &l
            )
            .total()
                > u.total()
        );
        assert_eq!(u.instances, 48);
    }

    #[test]
    fn precision_keeps_unit_models_binary_and_wide_models_int32() {
        for unit in [false, true] {
            assert_eq!(
                Precision::from_plan(
                    CoefficientPlan {
                        path: CoefficientPath::Wide,
                        scale_milli: 1
                    },
                    unit
                ),
                Precision::Int32
            );
        }
        let plan = CoefficientPlan {
            path: CoefficientPath::Int8,
            scale_milli: 1000,
        };
        assert_eq!(Precision::from_plan(plan, true), Precision::Binary);
        assert_eq!(Precision::from_plan(plan, false), Precision::Int8);
    }

    #[test]
    fn footprint_counts_three_csr_coefficient_slots_per_instance() {
        let l = a4000_limits();
        let shape = ModelShape {
            nodes: 100_000,
            couplers: 2_000_000,
            max_degree: 40,
        };
        let narrow = msa_footprint(shape, Precision::Int8, MsaStorage::Sparse, 128, 1, &l);
        let wide = msa_footprint(shape, Precision::Int32, MsaStorage::Sparse, 128, 1, &l);
        // Three slots, two directed halves per edge, three extra bytes per coefficient.
        assert_eq!(
            wide.per_instance_bytes - narrow.per_instance_bytes,
            36_900_000
        );
    }

    #[test]
    fn footprint_accounts_for_reads_and_spill_layout() {
        let l = a4000_limits();
        let shape = ModelShape {
            nodes: 100_000,
            couplers: 2_000_000,
            max_degree: 40,
        };
        let one = msa_footprint(shape, Precision::Binary, MsaStorage::Sparse, 64, 1, &l);
        let two = msa_footprint(shape, Precision::Binary, MsaStorage::Sparse, 128, 1, &l);
        // 64 more reads in three sample/energy slots, plus the larger spill.
        assert_eq!(two.per_instance_bytes - one.per_instance_bytes, 3_200_768);
        let mut no_shared = l;
        no_shared.shared_bytes_per_block_optin = 0;
        let spilled = msa_footprint(
            shape,
            Precision::Binary,
            MsaStorage::Sparse,
            128,
            1,
            &no_shared,
        );
        assert_eq!(spilled.per_instance_bytes - two.per_instance_bytes, 92_656);
    }

    #[test]
    fn dense_footprint_refuses_until_dense_storage_is_implemented() {
        let footprint = msa_footprint(
            ModelShape {
                nodes: 10,
                couplers: 45,
                max_degree: 9,
            },
            Precision::Float32,
            MsaStorage::Dense,
            128,
            1,
            &a4000_limits(),
        );
        assert_eq!(footprint.per_instance_bytes, usize::MAX);
        assert_eq!(footprint.total(), usize::MAX);
    }

    #[test]
    fn footprint_saturates_instead_of_wrapping_large_shapes() {
        let footprint = msa_footprint(
            ModelShape {
                nodes: usize::MAX,
                couplers: usize::MAX,
                max_degree: usize::MAX,
            },
            Precision::Int32,
            MsaStorage::Sparse,
            usize::MAX,
            1,
            &a4000_limits(),
        );
        assert_eq!(footprint.total(), usize::MAX);
        assert_eq!(
            Footprint {
                shared_bytes: 1,
                per_instance_bytes: usize::MAX / 2 + 1,
                instances: 2
            }
            .total(),
            usize::MAX
        );
    }

    #[test]
    fn footprint_counts_every_beta() {
        let limits = a4000_limits();
        let shape = ModelShape {
            nodes: 1000,
            couplers: 1000,
            max_degree: 2,
        };
        // build allocates an f32 beta; the integer lane adds i64 beta_fixed
        // and i32 delta_cut. All three tables are shared across instances.
        for (precision, bytes_per_beta) in [(Precision::Binary, 4), (Precision::Int32, 16)] {
            for betas in [1, 65_536] {
                let extra = 17;
                let base = msa_footprint(shape, precision, MsaStorage::Sparse, 128, betas, &limits);
                let longer = msa_footprint(
                    shape,
                    precision,
                    MsaStorage::Sparse,
                    128,
                    betas + extra,
                    &limits,
                );
                assert_eq!(longer.total() - base.total(), extra * bytes_per_beta);
                assert_eq!(longer.per_instance_bytes, base.per_instance_bytes);
            }
        }
    }

    #[test]
    fn footprint_beta_counts_use_standins_and_saturate() {
        let limits = a4000_limits();
        let shape = ModelShape {
            nodes: 1000,
            couplers: 1000,
            max_degree: 2,
        };
        for precision in [Precision::Binary, Precision::Int32] {
            let footprint =
                |betas| msa_footprint(shape, precision, MsaStorage::Sparse, 128, betas, &limits);
            assert_eq!(footprint(0), footprint(1));
            assert_eq!(footprint(usize::MAX).total(), usize::MAX);
        }
    }

    #[test]
    fn msa_budget_divides_usable_memory_between_instances() {
        assert_eq!(msa_per_instance_budget(&a4000_limits()), 268_435_456);
    }

    #[test]
    fn wide_budget_fits_the_compiled_frame_at_16_gib() {
        let limits = a4000(16 << 30);
        let frame_nodes = sa_budget(&limits);
        let wide = sa_wide_budget(frame_nodes, &limits);
        assert_eq!(wide, 36_069);
        assert!(sa_wide_working_set_bytes(wide, frame_nodes, &limits) <= limits.usable_bytes());
        assert!(
            sa_wide_working_set_bytes(wide + 1, frame_nodes, &limits) > limits.usable_bytes()
                || wide == frame_nodes
        );
    }

    #[test]
    fn wide_budget_fits_at_8_gib_without_a_rounding_gap() {
        let limits = a4000(8 << 30);
        let frame_nodes = sa_budget(&limits);
        let wide = sa_wide_budget(frame_nodes, &limits);
        assert_eq!(wide, 18_029);
        assert!(sa_wide_working_set_bytes(wide, frame_nodes, &limits) <= limits.usable_bytes());
        assert!(sa_wide_working_set_bytes(wide + 1, frame_nodes, &limits) > limits.usable_bytes());
    }

    #[test]
    fn wide_budget_covers_the_mining_frame() {
        let limits = a4000(16 << 30);
        assert_eq!(sa_wide_budget(4577, &limits), 4577);
        assert!(sa_wide_working_set_bytes(4577, 4577, &limits) <= limits.usable_bytes());
    }

    #[test]
    fn wide_budget_is_zero_when_the_frame_does_not_fit() {
        let limits = a4000(128 << 20);
        assert!(sa_wide_working_set_bytes(0, 4577, &limits) > limits.usable_bytes());
        assert_eq!(sa_wide_budget(4577, &limits), 0);
    }

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

    #[test]
    fn msa_spill_words_covers_every_nonce_past_the_shared_spins() {
        assert_eq!(msa_spill_words(5000, 5000, 2, 48), 0);
        assert_eq!(msa_spill_words(5000, 6000, 2, 48), 0);
        assert_eq!(
            msa_spill_words(200_000, 5791, 2, 48),
            (200_000 - 5791) * 2 * 48
        );
        assert_eq!(msa_spill_words(10, 0, 1, 1), 10);
    }

    /// An A4000 opts in to 101376 bytes (99 KB) per block. Less the two
    /// static ints and the 8704-byte threshold structures, two 8-byte
    /// words per spin fit 5791 spins.
    #[test]
    fn msa_shared_spins_on_a_99_kb_optin() {
        assert_eq!(msa_dynamic_shared_bytes(101_376), 101_360);
        assert_eq!(msa_shared_spins(101_360, MSA_REPLICA_WORDS), 5791);
        assert!(msa_shared_bytes(5791, MSA_REPLICA_WORDS) <= 101_360);
        assert!(msa_shared_bytes(5792, MSA_REPLICA_WORDS) > 101_360);
    }

    /// The other opt-in sizes in the supported-arch table. Volta (96 KB),
    /// Ampere datacenter (163 KB) and Hopper (227 KB) hold the default at two
    /// words. Turing (64 KB) does not.
    #[test]
    fn msa_shared_spins_across_the_supported_optin_sizes() {
        let two = MSA_REPLICA_WORDS;
        assert_eq!(
            msa_shared_spins(msa_dynamic_shared_bytes(98_304), two),
            5599
        );
        assert_eq!(
            msa_shared_spins(msa_dynamic_shared_bytes(166_912), two),
            9887
        );
        assert_eq!(
            msa_shared_spins(msa_dynamic_shared_bytes(232_448), two),
            13_983
        );
        assert_eq!(
            msa_shared_spins(msa_dynamic_shared_bytes(65_536), two),
            3551
        );
    }

    /// One word doubles the node budget, because the spin state is the only
    /// term that scales with the word count.
    #[test]
    fn msa_shared_spins_at_one_word_doubles_the_turing_ceiling() {
        assert_eq!(msa_shared_spins(msa_dynamic_shared_bytes(65_536), 1), 7102);
    }

    #[test]
    fn resolve_msa_is_bounded_by_the_kernel_ceiling_not_shared_memory() {
        // A Turing-sized opt-in holds 3551 spins at two words; the rest spill.
        let turing = DeviceLimits {
            shared_bytes_per_block_optin: 65_536,
            ..a4000(16 << 30)
        };
        assert_eq!(resolve(KernelKind::Msa, 8000, &turing), Ok(8000));
        assert_eq!(
            resolve(KernelKind::Msa, MSA_MAX_NODES, &turing),
            Ok(MSA_MAX_NODES)
        );
        assert_eq!(
            resolve(KernelKind::Msa, MSA_MAX_NODES + 1, &turing),
            Err(CapacityError::AboveDeviceBudget {
                requested: MSA_MAX_NODES + 1,
                budget: MSA_MAX_NODES,
                resource: BudgetResource::KernelCeiling,
            })
        );
    }

    /// Worst case is every spin spilled, one nonce per SM, so a card short of
    /// memory is refused with the node count its spill would fit.
    #[test]
    fn resolve_msa_refuses_a_spill_that_does_not_fit_device_memory() {
        let limits = a4000(64 << 20);
        let fits = |nodes: usize| {
            msa_spill_words(nodes, 0, MSA_REPLICA_WORDS, limits.sm_count) * 8
                <= limits.usable_bytes()
        };
        let Err(CapacityError::AboveDeviceBudget {
            requested,
            budget,
            resource,
        }) = resolve(KernelKind::Msa, MSA_MAX_NODES, &limits)
        else {
            panic!("64 MiB cannot hold the worst-case spill at 200k nodes");
        };
        assert_eq!(requested, MSA_MAX_NODES);
        assert_eq!(resource, BudgetResource::DeviceMemory);
        assert!(fits(budget), "budget {budget} must fit");
        assert!(!fits(budget + 1), "budget {budget} must be the largest");
        assert_eq!(resolve(KernelKind::Msa, budget, &limits), Ok(budget));
    }

    #[test]
    fn resolve_msa_accepts_the_ceiling_on_a_16_gib_card() {
        assert_eq!(
            resolve(KernelKind::Msa, MSA_MAX_NODES, &a4000(A4000_FREE)),
            Ok(MSA_MAX_NODES)
        );
    }

    #[test]
    fn advertised_msa_is_clamped_to_the_kernel_ceiling() {
        assert_eq!(advertised_nodes(KernelKind::Msa, 1_000_000), MSA_MAX_NODES);
    }

    #[test]
    fn kernel_ceiling_names_itself() {
        assert_eq!(BudgetResource::KernelCeiling.to_string(), "kernel-ceiling");
    }

    #[test]
    fn msa_read_cap_is_two_words_of_lanes() {
        assert_eq!(MSA_MAX_READS, 128);
    }

    #[test]
    fn advertised_passes_msa_through_between_the_default_and_the_ceiling() {
        assert_eq!(advertised_nodes(KernelKind::Msa, 8192), 8192);
        assert_eq!(advertised_nodes(KernelKind::Msa, 64), MSA_DEFAULT_NODES);
    }
}

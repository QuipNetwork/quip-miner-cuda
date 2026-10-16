//! Nonce-parallel probe screen.
//!
//! A `PoW` instance is drawn from its nonce, and the instance sets almost all of
//! the energy a solve can reach (quip-miner-metal
//! `docs/perf/2026-09-18-probe-screen-at-scale.md`). A short anneal therefore
//! ranks nonces, and a miner that spends its full solve only on the deepest
//! ranks finds valid proofs faster than one that solves every nonce.
//!
//! [`Screen`] runs that short anneal for many nonces at once
//! (`kernels/screen.cu`). Couplings are drawn on the device from each nonce
//! with the protocol's `ChaCha8` draw and scored on the device, so the host only
//! derives nonces and reads back one energy per lane. [`reference_masks`] and
//! [`reference_energy`] are the protocol-side references used by
//! `tests/screen_gpu.rs` to check the device.

use crate::cuda_device::{compile_for_arch, device_arch, driver_version, CudaDevice, CudaError};
use crate::jit_cache;
use crate::stream_pool::SessionStreams;
use crate::streaming::build_beta_schedule;
use crate::topology::SelfFeedingTopology;
use crate::IsingGraph;
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute_enum};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use quip_solver_core::{BetaSchedule, TopologyView};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SCREEN_SRC: &str = include_str!("../kernels/screen.cu");
/// Largest node degree the kernel handles (`SCR_MAX_DEG`).
pub const MAX_DEGREE: usize = 20;
/// One probe group is indivisible. This sweep cap bounds stop latency.
pub const MAX_PROBE_SWEEPS: usize = 2048;
const ROW: usize = 8192;
const CUT_ENTRIES: usize = 64;
const STATIC_SHARED_BYTES: usize = 128;

/// A reason to use ordinary MSA instead of the sign-only screen.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotScreenable {
    /// The screen requires exactly one allowed field, zero.
    #[error("screen fields must be [0]")]
    Fields,
    /// The screen requires both unit couplings, in either order.
    #[error("screen couplings must be [-1000, 1000] or [1000, -1000]")]
    Couplings,
    /// Packed node indices reserve 0xffff as a sentinel.
    #[error("screen node count {0} must be below 65535")]
    NodeCount(usize),
    /// Packed edge indices reserve 0xffff as a sentinel.
    #[error("screen edge count {0} must be below 65535")]
    EdgeCount(usize),
    /// An endpoint is outside the topology or an edge is a self-loop.
    #[error("screen edge ({u}, {v}) is invalid")]
    Edge {
        /// First dense endpoint.
        u: usize,
        /// Second dense endpoint.
        v: usize,
    },
    /// The padded adjacency row cannot hold this node's incident edges.
    #[error("screen node {node} has more than 20 incident edges")]
    Degree {
        /// Dense node index.
        node: usize,
    },
    /// Dynamic and static shared memory together exceed the opt-in limit.
    #[error("screen needs {need} B shared memory per block; device allows {have} B")]
    SharedMemory {
        /// Total bytes required.
        need: usize,
        /// Device opt-in limit.
        have: usize,
    },
    /// Probe dimensions cannot be represented by the kernel.
    #[error("screen requires reads_log2 2..=5, sweeps in 1..={}, and 32..=1024 threads in multiples of 32", MAX_PROBE_SWEEPS)]
    Shape,
    /// The device could not report its opt-in shared-memory limit.
    #[error("screen device budget unavailable: {0}")]
    Device(String),
}

/// Failures while opening or running a screen, separate from lease reporting.
#[derive(Debug, thiserror::Error)]
pub enum ScreenError {
    /// The caller should fall back to ordinary MSA.
    #[error(transparent)]
    NotScreenable(#[from] NotScreenable),
    /// A debug argument is outside the single-launch layout.
    #[error("bad screen parameter: {0}")]
    Param(String),
    /// Kernel compilation or module loading failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),
    /// A CUDA driver call failed.
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
}

/// Probe budget and block shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenParams {
    /// Reads per nonce = `1 << reads_log2`, with exponent 2..=5.
    pub reads_log2: u32,
    /// Sweeps per probe, in `1..=MAX_PROBE_SWEEPS`.
    pub sweeps: usize,
    /// Threads per block, a multiple of 32 in 32..=1024.
    pub threads: u32,
}

impl Default for ScreenParams {
    fn default() -> Self {
        Self {
            reads_log2: 2,
            sweeps: 512,
            threads: 256,
        }
    }
}

#[derive(Debug)]
struct ScreenTopology {
    edges: Vec<(usize, usize)>,
    /// `n * MAX_DEGREE` entries `neighbour << 16 | edge`, `u32::MAX` = none.
    adj: Vec<u32>,
    color_starts: Vec<i32>,
    color_counts: Vec<i32>,
    color_nodes: Vec<i32>,
    num_colors: i32,
    edge_u: Vec<i32>,
    edge_v: Vec<i32>,
    neg_bit: i32,
    unit_graph: IsingGraph,
}

/// Validated topology, probe parameters, and dynamic shared-memory requirement.
/// Produced only by [`ScreenDomain::check`].
#[derive(Debug)]
pub struct ScreenShape {
    topo: ScreenTopology,
    params: ScreenParams,
    shared: usize,
}

/// Admission to the screen's exact sign-only domain.
pub struct ScreenDomain;

impl ScreenDomain {
    /// Check the topology, probe parameters, and this device's opt-in budget.
    /// `TopologyView` has dense positions. Its protocol constructor rejects
    /// duplicate node identifiers before constructing the view.
    ///
    /// # Errors
    /// Returns the fallback reason without turning it into a wire error.
    pub fn check(
        device: &CudaDevice,
        topology: &TopologyView,
        params: ScreenParams,
    ) -> Result<ScreenShape, NotScreenable> {
        Self::check_with_budget(topology, params, shared_budget(device)?)
    }

    fn check_with_budget(
        topology: &TopologyView,
        params: ScreenParams,
        budget: usize,
    ) -> Result<ScreenShape, NotScreenable> {
        if !(2..=5).contains(&params.reads_log2)
            || params.sweeps == 0
            || params.sweeps > MAX_PROBE_SWEEPS
            || params.threads == 0
            || params.threads > 1024
            || !params.threads.is_multiple_of(32)
        {
            return Err(NotScreenable::Shape);
        }
        if topology.allowed_h_milli != [0] {
            return Err(NotScreenable::Fields);
        }
        let neg_bit = match topology.allowed_j_milli.as_slice() {
            [-1000, 1000] => 0,
            [1000, -1000] => 1,
            _ => return Err(NotScreenable::Couplings),
        };
        let n = topology.num_nodes;
        let edges = &topology.edges;
        if n >= 0xffff {
            return Err(NotScreenable::NodeCount(n));
        }
        if edges.len() >= 0xffff {
            return Err(NotScreenable::EdgeCount(edges.len()));
        }
        let shared = CUT_ENTRIES * 8 + n * 4 + edges.len().div_ceil(16) * 16 + ROW;
        check_shared(shared, budget)?;
        let mut adj = vec![u32::MAX; n * MAX_DEGREE];
        let mut degree = vec![0usize; n];
        for (k, &(u, v)) in edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                return Err(NotScreenable::Edge { u, v });
            }
            for (a, b) in [(u, v), (v, u)] {
                if degree[a] == MAX_DEGREE {
                    return Err(NotScreenable::Degree { node: a });
                }
                // Counts were checked below 0xffff before packing either id.
                let entry = (u32::try_from(b).unwrap() << 16) | u32::try_from(k).unwrap();
                adj[a * MAX_DEGREE + degree[a]] = entry;
                degree[a] += 1;
            }
        }
        let unit_graph = IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges.clone());
        let topo = SelfFeedingTopology::build(&unit_graph);
        Ok(ScreenShape {
            topo: ScreenTopology {
                edges: edges.clone(),
                adj,
                color_starts: topo.colors.starts,
                color_counts: topo.colors.counts,
                color_nodes: topo.colors.nodes,
                num_colors: topo.colors.num_colors,
                edge_u: edges
                    .iter()
                    .map(|&(u, _)| i32::try_from(u).unwrap())
                    .collect(),
                edge_v: edges
                    .iter()
                    .map(|&(_, v)| i32::try_from(v).unwrap())
                    .collect(),
                neg_bit,
                unit_graph,
            },
            params,
            shared,
        })
    }
}

fn shared_budget(device: &CudaDevice) -> Result<usize, NotScreenable> {
    device
        .ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        .map_err(|e| NotScreenable::Device(e.to_string()))
        .and_then(|v| usize::try_from(v).map_err(|e| NotScreenable::Device(e.to_string())))
}

fn check_shared(dynamic: usize, have: usize) -> Result<(), NotScreenable> {
    let need = dynamic + STATIC_SHARED_BYTES;
    if need > have {
        Err(NotScreenable::SharedMemory { need, have })
    } else {
        Ok(())
    }
}

/// One loaded screen using its device's context and a leased stream pair.
pub struct Screen<'a> {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    d_adj: CudaSlice<u32>,
    d_color_starts: CudaSlice<i32>,
    d_color_counts: CudaSlice<i32>,
    d_color_nodes: CudaSlice<i32>,
    d_edge_u: CudaSlice<i32>,
    d_edge_v: CudaSlice<i32>,
    d_cut: CudaSlice<u64>,
    num_colors: i32,
    num_betas: i32,
    sweeps_per_beta: i32,
    neg_bit: i32,
    n: i32,
    nnz: i32,
    max_groups: usize,
    shared: u32,
    params: ScreenParams,
    // Return the pair only after all device buffers have queued their frees.
    _streams: SessionStreams<'a>,
}

/// A complete run, or the completed prefix when the stop callback fired.
#[derive(Debug, PartialEq, Eq)]
pub struct ScreenRun {
    /// Exact milli energy per completed input nonce, in input order.
    pub best: Vec<i64>,
    /// True when a stop prevented the next batch from starting.
    pub stopped: bool,
}

/// Single-launch diagnostics for the protocol and threshold regression tests.
#[doc(hidden)]
pub struct ScreenDebug {
    /// Exact minima, excluding padded nonce slots.
    pub best: Vec<i64>,
    /// Final unit energy per lane.
    pub lanes: Vec<i32>,
    /// Drawn coupling sign masks per edge, in group order.
    pub masks: Vec<u32>,
    /// Final spin words per node, in group order.
    pub state: Vec<u32>,
    /// First beta's 64 thresholds copied from shared memory.
    pub cut: Vec<u64>,
}

impl<'a> Screen<'a> {
    /// Load through the JIT cache and upload a validated topology and schedule.
    /// The stream pool drains the pair before returning it, including on error.
    ///
    /// # Errors
    /// A different device may refuse the shape's shared memory. Compilation,
    /// module loading, allocation, and driver failures are returned separately.
    pub fn open(device: &'a CudaDevice, shape: ScreenShape) -> Result<Self, ScreenError> {
        let ScreenShape {
            topo,
            params,
            shared: need,
        } = shape;
        // A shape can be passed to another device after admission.
        check_shared(need, shared_budget(device)?)?;
        let sel = device_arch(&device.ctx);
        let arch = format!("sm_{sel}");
        let threads = format!("-DSCR_THREADS={}", params.threads);
        let defines = [threads.as_str()];
        let module = jit_cache::load_or_compile(
            &device.ctx,
            "screen",
            SCREEN_SRC,
            &defines,
            &arch,
            driver_version()?,
            topo.unit_graph.h.len(),
            || compile_for_arch(SCREEN_SRC, topo.unit_graph.h.len(), sel, &defines),
        )?;
        let func = module.load_function("quip_screen_probe")?;
        let static_shared = usize::try_from(func.shared_size_bytes()?)
            .map_err(|e| ScreenError::Param(e.to_string()))?;
        if static_shared > STATIC_SHARED_BYTES {
            return Err(ScreenError::Param(format!(
                "screen static shared memory {static_shared} exceeds {STATIC_SHARED_BYTES}"
            )));
        }
        func.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(need).map_err(|e| ScreenError::Param(e.to_string()))?,
        )?;
        let (schedule, sweeps_per_beta) = build_beta_schedule(
            &topo.unit_graph,
            params.sweeps,
            1,
            None,
            BetaSchedule::Geometric,
            1.0,
        );
        let cut = cut_table(&schedule);
        let streams = device.stream_pool.take();
        let stream = Arc::clone(&streams.compute);
        Ok(Self {
            d_adj: stream.clone_htod(&topo.adj)?,
            d_color_starts: stream.clone_htod(&topo.color_starts)?,
            d_color_counts: stream.clone_htod(&topo.color_counts)?,
            d_color_nodes: stream.clone_htod(&topo.color_nodes)?,
            d_edge_u: stream.clone_htod(&topo.edge_u)?,
            d_edge_v: stream.clone_htod(&topo.edge_v)?,
            d_cut: stream.clone_htod(&cut)?,
            num_colors: topo.num_colors,
            num_betas: i32::try_from(schedule.len())
                .map_err(|e| ScreenError::Param(e.to_string()))?,
            sweeps_per_beta: i32::try_from(sweeps_per_beta)
                .map_err(|e| ScreenError::Param(e.to_string()))?,
            neg_bit: topo.neg_bit,
            n: i32::try_from(topo.unit_graph.h.len())
                .map_err(|e| ScreenError::Param(e.to_string()))?,
            nnz: i32::try_from(topo.edges.len()).map_err(|e| ScreenError::Param(e.to_string()))?,
            max_groups: device.max_sms.max(1),
            shared: u32::try_from(need).map_err(|e| ScreenError::Param(e.to_string()))?,
            params,
            stream,
            func,
            _streams: streams,
        })
    }

    /// Parameters this screen was opened with.
    #[must_use]
    pub fn params(&self) -> ScreenParams {
        self.params
    }

    /// Screen arbitrary input lengths, checking `stop` before every launch.
    /// The first launch is one group. Later launches target 50 ms using twice
    /// the slowest observed wall time per group, capped at one group per SM.
    /// One group is the minimum cancellation granularity for this probe shape.
    ///
    /// # Errors
    /// Returns allocation or driver failures. Stops are successful partial runs.
    pub fn run(
        &self,
        nonces: &[[u8; 32]],
        stop: &dyn Fn() -> bool,
    ) -> Result<ScreenRun, ScreenError> {
        let mut best = Vec::new();
        let mut budget = LaunchBudget::new(self.max_groups);
        let npg = 32usize >> self.params.reads_log2;
        while best.len() < nonces.len() {
            if stop() {
                return Ok(ScreenRun {
                    best,
                    stopped: true,
                });
            }
            let end = nonces.len().min(best.len() + budget.groups() * npg);
            let batch = &nonces[best.len()..end];
            let start = Instant::now();
            let (lanes, _) = self.launch(batch, 7, false)?;
            budget.observe(batch.len().div_ceil(npg), start.elapsed());
            best.extend(best_milli(&lanes, batch.len(), self.params.reads_log2));
        }
        Ok(ScreenRun {
            best,
            stopped: false,
        })
    }

    /// Run one launch and return every group's state for protocol verification.
    /// The lane count is exactly 32 times the launched group count.
    ///
    /// # Errors
    /// Refuses empty inputs or more than one wave, and returns driver failures.
    #[doc(hidden)]
    pub fn run_debug(&self, nonces: &[[u8; 32]], seed: u32) -> Result<ScreenDebug, ScreenError> {
        let npg = 32usize >> self.params.reads_log2;
        if nonces.is_empty() || nonces.len().div_ceil(npg) > self.max_groups {
            return Err(ScreenError::Param(
                "debug requires one nonempty wave".into(),
            ));
        }
        let (lanes, debug) = self.launch(nonces, seed, true)?;
        let (masks, state, cut) = debug.unwrap();
        Ok(ScreenDebug {
            best: best_milli(&lanes, nonces.len(), self.params.reads_log2),
            lanes,
            masks,
            state,
            cut,
        })
    }

    #[expect(
        clippy::type_complexity,
        reason = "the optional tuple holds the kernel's three debug buffers in argument order"
    )]
    fn launch(
        &self,
        nonces: &[[u8; 32]],
        seed: u32,
        debug: bool,
    ) -> Result<(Vec<i32>, Option<(Vec<u32>, Vec<u32>, Vec<u64>)>), ScreenError> {
        let npg = 32usize >> self.params.reads_log2;
        let groups = nonces.len().div_ceil(npg);
        let mut keys = vec![0u32; groups * npg * 8];
        for (slot, nonce) in keys.chunks_exact_mut(8).zip(nonces) {
            for (word, bytes) in slot.iter_mut().zip(nonce.chunks_exact(4)) {
                *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
        // Per-launch buffers keep run(&self) independent of mutable session state.
        let d_keys = self.stream.clone_htod(&keys)?;
        let mut d_out = self.stream.alloc_zeros::<i32>(groups * 32)?;
        let mut dbg = if debug {
            Some((
                self.stream
                    .alloc_zeros::<u32>(groups * usize::try_from(self.nnz).unwrap())?,
                self.stream
                    .alloc_zeros::<u32>(groups * usize::try_from(self.n).unwrap())?,
                self.stream.alloc_zeros::<u64>(CUT_ENTRIES)?,
            ))
        } else {
            None
        };
        let null: u64 = 0;
        let reads_log2 = i32::try_from(self.params.reads_log2).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (u32::try_from(groups).unwrap(), 1, 1),
            block_dim: (self.params.threads, 1, 1),
            shared_mem_bytes: self.shared,
        };
        {
            let mut b = self.stream.launch_builder(&self.func);
            b.arg(&d_keys)
                .arg(&reads_log2)
                .arg(&self.neg_bit)
                .arg(&self.n)
                .arg(&self.nnz)
                .arg(&self.d_adj)
                .arg(&self.d_color_starts)
                .arg(&self.d_color_counts)
                .arg(&self.d_color_nodes)
                .arg(&self.num_colors)
                .arg(&self.d_edge_u)
                .arg(&self.d_edge_v)
                .arg(&self.d_cut)
                .arg(&self.num_betas)
                .arg(&self.sweeps_per_beta)
                .arg(&seed)
                .arg(&mut d_out);
            match dbg.as_mut() {
                Some((mask, state, cut)) => b.arg(mask).arg(state).arg(cut),
                None => b.arg(&null).arg(&null).arg(&null),
            };
            // SAFETY: argument types and order match quip_screen_probe. Buffers
            // cover every launched group, node, edge, and threshold, including
            // padded nonce slots. Optional debug pointers are checked by the
            // kernel. Synchronous readback keeps all buffers alive until completion.
            unsafe { b.launch(cfg) }?;
        }
        let lanes = self.stream.clone_dtoh(&d_out)?;
        let debug = match dbg {
            Some((mask, state, cut)) => Some((
                self.stream.clone_dtoh(&mask)?,
                self.stream.clone_dtoh(&state)?,
                self.stream.clone_dtoh(&cut)?,
            )),
            None => None,
        };
        Ok((lanes, debug))
    }
}

struct LaunchBudget {
    max_groups: usize,
    nanos_per_group: u128,
}

impl LaunchBudget {
    fn new(max_groups: usize) -> Self {
        Self {
            max_groups,
            nanos_per_group: 0,
        }
    }

    fn groups(&self) -> usize {
        if self.nanos_per_group == 0 {
            return 1;
        }
        let groups = 50_000_000 / self.nanos_per_group / 2;
        usize::try_from(groups)
            .unwrap_or(self.max_groups)
            .clamp(1, self.max_groups)
    }

    fn observe(&mut self, groups: usize, elapsed: Duration) {
        self.nanos_per_group = self
            .nanos_per_group
            .max(elapsed.as_nanos().div_ceil(groups as u128).max(1));
    }
}

fn best_milli(lanes: &[i32], count: usize, reads_log2: u32) -> Vec<i64> {
    let npg = 32usize >> reads_log2;
    (0..count)
        .map(|i| {
            let base = (i / npg) * 32;
            let best = lanes[base..base + 32]
                .iter()
                .skip(i % npg)
                .step_by(npg)
                .copied()
                .min()
                .unwrap();
            i64::from(best) * 1000
        })
        .collect()
}

/// Metropolis thresholds per rung, the arithmetic of the msa kernel's table:
/// f32 beta widened to f64, p = exp(-2 beta m), cut = p * 2^64, all ones for
/// p >= 1.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "p is in (0, 1) on that branch, so p * 2^64 is a non-negative value below 2^64"
)]
fn cut_table(schedule: &[f32]) -> Vec<u64> {
    let mut cut = Vec::with_capacity(schedule.len() * CUT_ENTRIES);
    for &b in schedule {
        let beta = f64::from(b);
        for m in 0..64u32 {
            let p = (-2.0 * beta * f64::from(m)).exp();
            cut.push(if p >= 1.0 {
                u64::MAX
            } else {
                (p * 18_446_744_073_709_551_616.0) as u64
            });
        }
    }
    cut
}

/// Protocol coupling signs for the first group, in original edge order.
///
/// # Errors
/// Returns a protocol draw error for invalid allowed tables.
pub fn reference_masks(
    nonces: &[[u8; 32]],
    reads_log2: u32,
    topo: &TopologyView,
) -> Result<Vec<u32>, ScreenError> {
    let npg = 32usize >> reads_log2;
    let mut masks = vec![0u32; topo.edges.len()];
    for lane in 0..32usize {
        let Some(nonce) = nonces.get(lane % npg) else {
            continue;
        };
        let (_, j) = topo
            .draw(*nonce)
            .map_err(|e| ScreenError::Param(e.to_string()))?;
        for (mask, &v) in masks.iter_mut().zip(&j) {
            if v < 0 {
                *mask |= 1 << lane;
            }
        }
    }
    Ok(masks)
}

/// Exact protocol milli energy of one debug lane's final spins.
///
/// # Errors
/// Returns a protocol draw error for invalid allowed tables.
pub fn reference_energy(
    nonce: [u8; 32],
    state: &[u32],
    lane: usize,
    topo: &TopologyView,
) -> Result<i64, ScreenError> {
    let (h, j) = topo
        .draw(nonce)
        .map_err(|e| ScreenError::Param(e.to_string()))?;
    let spins: Vec<i8> = state
        .iter()
        .map(|w| if (w >> lane) & 1 == 1 { -1 } else { 1 })
        .collect();
    let h: Vec<f64> = h.iter().map(|&v| f64::from(v) / 1000.0).collect();
    let j: Vec<f64> = j.iter().map(|&v| f64::from(v) / 1000.0).collect();
    Ok(quip_protocol::scoring::energy_milli(
        &spins,
        &h,
        &j,
        &topo.edges,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> TopologyView {
        TopologyView {
            num_nodes: n,
            edges: (0..n).map(|i| (i, (i + 1) % n)).collect(),
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        }
    }

    fn check(topology: &TopologyView) -> Result<ScreenShape, NotScreenable> {
        ScreenDomain::check_with_budget(topology, ScreenParams::default(), usize::MAX)
    }

    #[test]
    fn every_row_entry_points_back_to_its_original_edge() {
        let mut view = ring(6);
        view.edges.rotate_left(3);
        let shape = check(&view).unwrap();
        assert_eq!(shape.topo.edges, view.edges);
        for v in 0..6 {
            let row = &shape.topo.adj[v * MAX_DEGREE..(v + 1) * MAX_DEGREE];
            let used: Vec<u32> = row.iter().copied().filter(|&e| e != u32::MAX).collect();
            assert_eq!(used.len(), 2);
            for e in used {
                let (nb, k) = ((e >> 16) as usize, (e & 0xffff) as usize);
                let (a, b) = view.edges[k];
                assert!((a, b) == (v, nb) || (a, b) == (nb, v));
            }
        }
    }

    #[test]
    fn both_unit_coupling_orders_are_screenable() {
        for (couplings, neg_bit) in [([-1000, 1000], 0), ([1000, -1000], 1)] {
            let mut view = ring(4);
            view.allowed_j_milli = couplings.to_vec();
            assert_eq!(check(&view).unwrap().topo.neg_bit, neg_bit);
        }
    }

    #[test]
    fn non_unit_couplings_are_not_screenable() {
        for couplings in [
            vec![-2000, 2000],
            vec![-1000, 2000],
            vec![-1000, 0, 1000],
            vec![],
            vec![1000, 1000],
        ] {
            let mut view = ring(4);
            view.allowed_j_milli = couplings;
            assert!(matches!(check(&view), Err(NotScreenable::Couplings)));
        }
    }

    #[test]
    fn only_a_single_zero_field_is_screenable() {
        for fields in [vec![1], vec![-1000, 0, 1000], vec![], vec![0, 0]] {
            let mut view = ring(4);
            view.allowed_h_milli = fields;
            assert!(matches!(check(&view), Err(NotScreenable::Fields)));
        }
    }

    #[test]
    fn packed_counts_must_be_below_65535() {
        let mut view = ring(4);
        view.num_nodes = 65535;
        assert!(matches!(check(&view), Err(NotScreenable::NodeCount(65535))));
        view.num_nodes = 65534;
        assert!(check(&view).is_ok());
        view.edges = vec![(0, 1); 65535];
        assert!(matches!(check(&view), Err(NotScreenable::EdgeCount(65535))));
        // A 65,534-edge ring has degree two, so the edge boundary fits.
        view.edges = (0..65534).map(|i| (i, (i + 1) % 65534)).collect();
        assert!(check(&view).is_ok());
    }

    #[test]
    fn invalid_endpoints_and_self_loops_are_not_screenable() {
        for edge in [(0, 4), (4, 0), (2, 2)] {
            let mut view = ring(4);
            view.edges = vec![edge];
            assert!(matches!(check(&view), Err(NotScreenable::Edge { .. })));
        }
    }

    #[test]
    fn degree_twenty_fits_but_twenty_one_does_not() {
        let mut view = ring(22);
        view.edges = (1..=20).map(|v| (0, v)).collect();
        assert!(check(&view).is_ok());
        view.edges.push((0, 21));
        assert!(matches!(
            check(&view),
            Err(NotScreenable::Degree { node: 0 })
        ));
    }

    #[test]
    fn shared_budget_includes_padding_and_static_memory() {
        let view = ring(6);
        // 512 cut + 24 state + 16 padded signs + 8192 row + 128 lane counts.
        let need = 8872;
        let params = ScreenParams::default();
        assert!(ScreenDomain::check_with_budget(&view, params, need).is_ok());
        assert!(matches!(
            ScreenDomain::check_with_budget(&view, params, need - 1),
            Err(NotScreenable::SharedMemory {
                need: 8872,
                have: 8871
            })
        ));
    }

    #[test]
    fn probe_sweep_limit_bounds_indivisible_work() {
        let params = ScreenParams {
            sweeps: 2048,
            ..ScreenParams::default()
        };
        assert!(ScreenDomain::check_with_budget(&ring(4), params, usize::MAX).is_ok());
        let reason = ScreenDomain::check_with_budget(
            &ring(4),
            ScreenParams {
                sweeps: 2049,
                ..params
            },
            usize::MAX,
        )
        .unwrap_err();
        assert_eq!(reason, NotScreenable::Shape);
        assert!(reason.to_string().contains("2048"));
    }

    #[test]
    fn invalid_probe_shapes_are_not_screenable() {
        let params = ScreenParams::default();
        for shape in [
            ScreenParams {
                reads_log2: 1,
                ..params
            },
            ScreenParams {
                reads_log2: 6,
                ..params
            },
            ScreenParams {
                sweeps: 0,
                ..params
            },
            ScreenParams {
                sweeps: usize::MAX,
                ..params
            },
            ScreenParams {
                threads: 0,
                ..params
            },
            ScreenParams {
                threads: 33,
                ..params
            },
            ScreenParams {
                threads: 1056,
                ..params
            },
        ] {
            assert!(matches!(
                ScreenDomain::check_with_budget(&ring(4), shape, usize::MAX),
                Err(NotScreenable::Shape)
            ));
        }
    }

    #[test]
    fn milli_minima_use_nonce_lane_layout_and_exclude_padding() {
        for reads_log2 in 2..=5 {
            let npg = 32usize >> reads_log2;
            let mut lanes = vec![100; 64];
            for nonce in 0..=npg {
                let base = (nonce / npg) * 32;
                lanes[base + nonce % npg + npg] = -i32::try_from(nonce + 1).unwrap();
            }
            // A padded nonce has a deeper result than any real input.
            if npg > 1 {
                lanes[33] = i32::MIN;
            }
            let got = best_milli(&lanes, npg + 1, reads_log2);
            let want: Vec<i64> = (1..=npg + 1)
                .map(|i| -i64::try_from(i).unwrap() * 1000)
                .collect();
            assert_eq!(got, want);
        }
        assert_eq!(best_milli(&[i32::MIN; 32], 1, 5), vec![-2_147_483_648_000]);
        assert!(best_milli(&[0; 32], 0, 2).is_empty());
    }

    #[test]
    fn launch_budget_starts_small_and_reduces_after_slow_work() {
        let mut budget = LaunchBudget::new(80);
        assert_eq!(budget.groups(), 1);
        budget.observe(1, std::time::Duration::from_millis(1));
        assert_eq!(budget.groups(), 25);
        budget.observe(5, std::time::Duration::from_millis(250));
        assert_eq!(budget.groups(), 1);
        // Keep the slowest observed cost even after a faster batch.
        budget.observe(1, std::time::Duration::from_micros(1));
        assert_eq!(budget.groups(), 1);
        let mut budget = LaunchBudget::new(2);
        budget.observe(1, std::time::Duration::from_micros(1));
        assert_eq!(budget.groups(), 2);
    }

    #[test]
    fn cut_table_is_all_ones_at_m_zero_and_decreasing() {
        let cut = cut_table(&[0.5, 2.0]);
        assert_eq!(cut.len(), 128);
        for rung in cut.chunks_exact(64) {
            assert_eq!(rung[0], u64::MAX);
            assert!(rung.windows(2).all(|w| w[1] <= w[0]));
        }
    }
}

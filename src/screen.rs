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
//! [`reference_energy`] are the protocol-side references `quip-screen verify`
//! checks the device against.

use crate::cuda_device::select_arch;
use crate::streaming::build_beta_schedule;
use crate::topology::SelfFeedingTopology;
use crate::IsingGraph;
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute_enum};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

const SCREEN_SRC: &str = include_str!("../kernels/screen.cu");
/// Largest node degree the kernel handles (`SCR_MAX_DEG`).
pub const MAX_DEGREE: usize = 20;
const ROW: usize = 8192;
const CUT_ENTRIES: usize = 64;

/// Why a screen could not be built or run.
#[derive(Debug, thiserror::Error)]
pub enum ScreenError {
    /// The topology or its allowed values are outside what the kernel handles.
    #[error("unsupported topology: {0}")]
    Topology(String),
    /// A launch parameter is out of range.
    #[error("bad parameter: {0}")]
    Param(String),
    /// The device cannot hold one group's shared memory.
    #[error("screen needs {need} B of shared memory per block; device {device} allows {have} B")]
    SharedMemory {
        /// Bytes one block needs.
        need: usize,
        /// Opt-in bytes per block the device allows.
        have: usize,
        /// CUDA device index.
        device: usize,
    },
    /// NVRTC failed to compile the kernel.
    #[error("NVRTC: {0}")]
    Compile(String),
    /// A CUDA driver call failed.
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
}

/// A topology packed for the screen kernel: one padded neighbour row per node
/// (neighbour and edge index per entry), colour classes, and the endpoints of
/// every edge.
pub struct ScreenTopology {
    n: usize,
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
    /// Unit couplings on this graph. The default beta range depends on |J|
    /// only, so every instance of the topology shares it.
    unit_graph: IsingGraph,
}

impl ScreenTopology {
    /// Pack `edges` over `n` nodes (dense ids, the draw order of the chain's
    /// topology) with the allowed field and coupling values.
    ///
    /// # Errors
    /// [`ScreenError::Topology`] unless every field is 0, the couplings are one
    /// negative and one positive value, no node exceeds [`MAX_DEGREE`], and
    /// every edge is inside `0..n` with no self-loops, with fewer than 65,535
    /// nodes and edges (the packed rows hold 16-bit ids).
    pub fn new(
        n: usize,
        edges: Vec<(usize, usize)>,
        fields_milli: &[i32],
        couplings_milli: &[i32],
    ) -> Result<Self, ScreenError> {
        if fields_milli != [0] {
            return Err(ScreenError::Topology(format!(
                "fields must be {{0}}, got {fields_milli:?}"
            )));
        }
        let neg_bit = match couplings_milli {
            [a, b] if *a < 0 && *b > 0 => 0,
            [a, b] if *a > 0 && *b < 0 => 1,
            _ => {
                return Err(ScreenError::Topology(format!(
                    "couplings must be one negative and one positive value, got {couplings_milli:?}"
                )))
            }
        };
        if let Some(&(u, v)) = edges.iter().find(|&&(u, v)| u >= n || v >= n || u == v) {
            return Err(ScreenError::Topology(format!(
                "edge ({u}, {v}) with {n} nodes"
            )));
        }
        let to_i32 = |v: usize| {
            i32::try_from(v).map_err(|_| ScreenError::Topology(format!("{v} does not fit i32")))
        };
        if n >= 0xffff || edges.len() >= 0xffff {
            return Err(ScreenError::Topology(format!(
                "{n} nodes and {} edges; the packed rows hold 16-bit ids",
                edges.len()
            )));
        }
        let unit_graph = IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges.clone());
        let topo = SelfFeedingTopology::build(&unit_graph);
        let mut adj = vec![u32::MAX; n * MAX_DEGREE];
        let mut degree = vec![0usize; n];
        for (k, &(u, v)) in edges.iter().enumerate() {
            for (a, b) in [(u, v), (v, u)] {
                if degree[a] == MAX_DEGREE {
                    return Err(ScreenError::Topology(format!(
                        "node {a} has more than {MAX_DEGREE} neighbours"
                    )));
                }
                // Both ids are below 0xffff (checked above), so the casts are exact.
                let entry = (u32::try_from(b).unwrap_or(u32::MAX) << 16)
                    | u32::try_from(k).unwrap_or(u32::MAX);
                adj[a * MAX_DEGREE + degree[a]] = entry;
                degree[a] += 1;
            }
        }
        let edge_u = edges
            .iter()
            .map(|&(u, _)| to_i32(u))
            .collect::<Result<_, _>>()?;
        let edge_v = edges
            .iter()
            .map(|&(_, v)| to_i32(v))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            n,
            adj,
            color_starts: topo.colors.starts,
            color_counts: topo.colors.counts,
            color_nodes: topo.colors.nodes,
            num_colors: topo.colors.num_colors,
            edges,
            edge_u,
            edge_v,
            neg_bit,
            unit_graph,
        })
    }

    /// Node count.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.n
    }

    /// Edges in draw order.
    #[must_use]
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Shared memory one group needs, in bytes.
    #[must_use]
    pub fn shared_bytes(&self) -> usize {
        CUT_ENTRIES * 8 + self.n * 4 + self.edges.len().div_ceil(16) * 16 + ROW
    }
}

/// Probe budget and launch shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenParams {
    /// Reads per nonce = `1 << reads_log2`, 2..=5 (4 to 32 reads; a group of
    /// 32 lanes then holds 8 down to 1 nonces).
    pub reads_log2: u32,
    /// Sweeps per probe.
    pub sweeps: usize,
    /// Groups per launch = streaming multiprocessors x `waves`.
    pub waves: usize,
    /// Threads per block (a multiple of 32, at most 1024).
    pub threads: u32,
}

impl Default for ScreenParams {
    fn default() -> Self {
        Self {
            reads_log2: 2,
            sweeps: 512,
            waves: 16,
            threads: 256,
        }
    }
}

/// One loaded probe screen: a compiled kernel and device-resident topology.
pub struct Screen {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    d_adj: CudaSlice<u32>,
    d_color_starts: CudaSlice<i32>,
    d_color_counts: CudaSlice<i32>,
    d_color_nodes: CudaSlice<i32>,
    d_edge_u: CudaSlice<i32>,
    d_edge_v: CudaSlice<i32>,
    d_cut: CudaSlice<u64>,
    d_keys: CudaSlice<u32>,
    d_out: CudaSlice<i32>,
    num_colors: i32,
    num_betas: i32,
    sweeps_per_beta: i32,
    neg_bit: i32,
    n: i32,
    nnz: i32,
    groups: u32,
    shared: u32,
    params: ScreenParams,
}

/// Device output of one launch.
pub struct ScreenRun {
    /// Best energy (energy units) per input nonce, in input order.
    pub best: Vec<i32>,
    /// Energy of every lane of group 0 (lane L is read L / npg of nonce L % npg).
    pub group0_lanes: Vec<i32>,
    /// With `debug`: group 0's lane masks per edge and final spin words.
    pub group0_debug: Option<(Vec<u32>, Vec<u32>)>,
}

impl Screen {
    /// Compile the kernel for `device` and upload the topology and schedule.
    ///
    /// # Errors
    /// Bad parameters, a device without enough shared memory per block, or a
    /// compile or driver failure.
    pub fn open(
        device: usize,
        topo: &ScreenTopology,
        params: ScreenParams,
    ) -> Result<Self, ScreenError> {
        if !(2..=5).contains(&params.reads_log2) {
            return Err(ScreenError::Param("reads_log2 must be 2..=5".into()));
        }
        if params.threads == 0 || params.threads > 1024 || !params.threads.is_multiple_of(32) {
            return Err(ScreenError::Param(
                "threads must be a multiple of 32, at most 1024".into(),
            ));
        }
        if params.sweeps == 0 || params.waves == 0 {
            return Err(ScreenError::Param(
                "sweeps and waves must be positive".into(),
            ));
        }
        let ctx = CudaContext::new(device)?;
        let stream = ctx.default_stream();
        let cc = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?
            * 10
            + ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let sms = usize::try_from(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )
        .unwrap_or(1)
        .max(1);
        let have = usize::try_from(ctx.attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )?)
        .unwrap_or(0);
        let need = topo.shared_bytes();
        // The kernel's static `lane_count[32]` shares the same budget.
        if need + 128 > have {
            return Err(ScreenError::SharedMemory { need, have, device });
        }
        let opts = CompileOptions {
            use_fast_math: Some(true),
            options: vec![
                format!("--gpu-architecture=compute_{}", select_arch(cc)),
                format!("-DSCR_THREADS={}", params.threads),
            ],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(SCREEN_SRC, opts)
            .map_err(|e| ScreenError::Compile(e.to_string()))?;
        let func = ctx.load_module(ptx)?.load_function("quip_screen_probe")?;
        let shared =
            u32::try_from(need).map_err(|_| ScreenError::Param("shared memory size".into()))?;
        func.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(need).map_err(|_| ScreenError::Param("shared memory size".into()))?,
        )?;

        let (schedule, sweeps_per_beta) =
            build_beta_schedule(&topo.unit_graph, params.sweeps, 1, None);
        let cut = cut_table(&schedule);
        let groups = sms
            .checked_mul(params.waves)
            .and_then(|g| u32::try_from(g).ok())
            .ok_or_else(|| ScreenError::Param("too many groups".into()))?;
        let npg = 32usize >> params.reads_log2;
        let to_i32 = |v: usize, what: &str| {
            i32::try_from(v).map_err(|_| ScreenError::Param(format!("{what} does not fit i32")))
        };
        Ok(Self {
            d_adj: stream.clone_htod(&topo.adj)?,
            d_color_starts: stream.clone_htod(&topo.color_starts)?,
            d_color_counts: stream.clone_htod(&topo.color_counts)?,
            d_color_nodes: stream.clone_htod(&topo.color_nodes)?,
            d_edge_u: stream.clone_htod(&topo.edge_u)?,
            d_edge_v: stream.clone_htod(&topo.edge_v)?,
            d_cut: stream.clone_htod(&cut)?,
            d_keys: stream.alloc_zeros::<u32>(groups as usize * npg * 8)?,
            d_out: stream.alloc_zeros::<i32>(groups as usize * 32)?,
            num_colors: topo.num_colors,
            num_betas: to_i32(schedule.len(), "beta count")?,
            sweeps_per_beta: to_i32(sweeps_per_beta, "sweeps per beta")?,
            neg_bit: topo.neg_bit,
            n: to_i32(topo.n, "node count")?,
            nnz: to_i32(topo.edges.len(), "edge count")?,
            groups,
            shared,
            params,
            stream,
            func,
        })
    }

    /// Parameters this screen was opened with.
    #[must_use]
    pub fn params(&self) -> ScreenParams {
        self.params
    }

    /// Nonces screened per launch.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.groups as usize * (32 >> self.params.reads_log2)
    }

    /// Screen up to [`Screen::batch`] nonces. Unused slots run on zero keys and
    /// are ignored.
    ///
    /// # Errors
    /// [`ScreenError::Param`] when `nonces` exceeds the batch, or a driver
    /// failure.
    pub fn run(
        &mut self,
        nonces: &[[u8; 32]],
        seed: u32,
        debug: bool,
    ) -> Result<ScreenRun, ScreenError> {
        let batch = self.batch();
        if nonces.len() > batch {
            return Err(ScreenError::Param(format!(
                "{} nonces for a batch of {batch}",
                nonces.len()
            )));
        }
        let mut keys = vec![0u32; batch * 8];
        for (slot, nonce) in keys.chunks_exact_mut(8).zip(nonces) {
            for (word, bytes) in slot.iter_mut().zip(nonce.chunks_exact(4)) {
                *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
        self.stream.memcpy_htod(keys.as_slice(), &mut self.d_keys)?;
        let nnz = usize::try_from(self.nnz).unwrap_or(0);
        let n = usize::try_from(self.n).unwrap_or(0);
        let mut dbg = if debug {
            Some((
                self.stream.alloc_zeros::<u32>(nnz)?,
                self.stream.alloc_zeros::<u32>(n)?,
            ))
        } else {
            None
        };
        let null: u64 = 0;
        let reads_log2 = i32::try_from(self.params.reads_log2).unwrap_or(2);
        let cfg = LaunchConfig {
            grid_dim: (self.groups, 1, 1),
            block_dim: (self.params.threads, 1, 1),
            shared_mem_bytes: self.shared,
        };
        {
            let mut b = self.stream.launch_builder(&self.func);
            b.arg(&self.d_keys)
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
                .arg(&mut self.d_out);
            match dbg.as_mut() {
                Some((mask, state)) => b.arg(mask).arg(state),
                None => b.arg(&null).arg(&null),
            };
            // SAFETY: the argument list matches `quip_screen_probe` in
            // kernels/screen.cu in order and type: device slices for every
            // pointer (a 0 u64 for the optional debug pointers, which the
            // kernel checks), i32/u32 scalars for the rest. Every buffer is
            // sized from this screen's own `groups`, `npg`, `n` and `nnz`,
            // which the kernel indexes within, and all of them outlive the
            // launch because `clone_dtoh` below synchronizes the stream first.
            unsafe { b.launch(cfg) }?;
        }
        let lanes: Vec<i32> = self.stream.clone_dtoh(&self.d_out)?;
        let npg = 32usize >> self.params.reads_log2;
        let best = (0..nonces.len())
            .map(|i| {
                let base = (i / npg) * 32;
                lanes[base..base + 32]
                    .iter()
                    .skip(i % npg)
                    .step_by(npg)
                    .copied()
                    .min()
                    .unwrap_or(i32::MAX)
            })
            .collect();
        let group0_debug = match dbg {
            Some((mask, state)) => Some((
                self.stream.clone_dtoh(&mask)?,
                self.stream.clone_dtoh(&state)?,
            )),
            None => None,
        };
        Ok(ScreenRun {
            best,
            group0_lanes: lanes[..32].to_vec(),
            group0_debug,
        })
    }
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

/// The protocol's own draw for group 0 of a launch, as lane masks per edge:
/// bit L is set when nonce `L % npg` has a negative coupling on that edge.
///
/// # Errors
/// [`ScreenError::Topology`] if the protocol draw rejects the allowed values.
pub fn reference_masks(
    nonces: &[[u8; 32]],
    reads_log2: u32,
    topo: &ScreenTopology,
    allowed_j_milli: &[i32],
) -> Result<Vec<u32>, ScreenError> {
    let npg = 32usize >> reads_log2;
    let mut masks = vec![0u32; topo.edges.len()];
    for lane in 0..32usize {
        let Some(nonce) = nonces.get(lane % npg) else {
            continue;
        };
        let (_, j) = quip_protocol::chacha8::draw_ising_milli(
            *nonce,
            topo.n,
            topo.edges.len(),
            &[0],
            allowed_j_milli,
        )
        .map_err(|e| ScreenError::Topology(e.to_string()))?;
        for (mask, &v) in masks.iter_mut().zip(&j) {
            if v < 0 {
                *mask |= 1 << lane;
            }
        }
    }
    Ok(masks)
}

/// Energy (units) of lane `lane`'s spins under `nonce`'s instance, through the
/// protocol's scoring function. Bit set = spin -1, as in the other kernels.
///
/// # Errors
/// [`ScreenError::Topology`] if the protocol draw rejects the allowed values.
pub fn reference_energy(
    nonce: [u8; 32],
    state: &[u32],
    lane: usize,
    topo: &ScreenTopology,
    allowed_j_milli: &[i32],
) -> Result<i64, ScreenError> {
    let (h, j) = quip_protocol::chacha8::draw_ising_milli(
        nonce,
        topo.n,
        topo.edges.len(),
        &[0],
        allowed_j_milli,
    )
    .map_err(|e| ScreenError::Topology(e.to_string()))?;
    let spins: Vec<i8> = state
        .iter()
        .map(|w| if (w >> lane) & 1 == 1 { -1 } else { 1 })
        .collect();
    let h: Vec<f64> = h.iter().map(|&v| f64::from(v) / 1000.0).collect();
    let j: Vec<f64> = j.iter().map(|&v| f64::from(v) / 1000.0).collect();
    Ok(quip_protocol::scoring::energy_milli(&spins, &h, &j, &topo.edges) / 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> Vec<(usize, usize)> {
        (0..n).map(|i| (i, (i + 1) % n)).collect()
    }

    #[test]
    fn every_row_entry_points_back_to_its_edge() {
        let topo = ScreenTopology::new(6, ring(6), &[0], &[-1000, 1000]).unwrap();
        for v in 0..6 {
            let row = &topo.adj[v * MAX_DEGREE..(v + 1) * MAX_DEGREE];
            let used: Vec<u32> = row.iter().copied().filter(|&e| e != u32::MAX).collect();
            assert_eq!(used.len(), 2, "ring node {v} has two neighbours");
            for e in used {
                let (nb, k) = ((e >> 16) as usize, (e & 0xffff) as usize);
                let (a, b) = topo.edges[k];
                assert!(
                    (a, b) == (v, nb) || (a, b) == (nb, v),
                    "entry {e:#x} of node {v}"
                );
            }
        }
    }

    #[test]
    fn coupling_order_sets_the_negative_bit() {
        assert_eq!(
            ScreenTopology::new(4, ring(4), &[0], &[-1000, 1000])
                .unwrap()
                .neg_bit,
            0
        );
        assert_eq!(
            ScreenTopology::new(4, ring(4), &[0], &[1000, -1000])
                .unwrap()
                .neg_bit,
            1
        );
    }

    #[test]
    fn unsupported_values_and_graphs_are_refused() {
        assert!(ScreenTopology::new(4, ring(4), &[-1000, 0, 1000], &[-1000, 1000]).is_err());
        assert!(ScreenTopology::new(4, ring(4), &[0], &[-1000, 0, 1000]).is_err());
        assert!(ScreenTopology::new(4, vec![(0, 4)], &[0], &[-1000, 1000]).is_err());
        let star: Vec<(usize, usize)> = (1..=MAX_DEGREE + 1).map(|v| (0, v)).collect();
        assert!(ScreenTopology::new(MAX_DEGREE + 2, star, &[0], &[-1000, 1000]).is_err());
    }

    #[test]
    fn shared_budget_matches_the_kernel_layout() {
        let topo = ScreenTopology::new(6, ring(6), &[0], &[-1000, 1000]).unwrap();
        assert_eq!(topo.shared_bytes(), 64 * 8 + 6 * 4 + 16 + 8192);
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

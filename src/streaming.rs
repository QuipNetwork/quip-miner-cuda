//! Self-feeding streaming session: persistent kernel + 3-slot rotation.
//!
//! Rust port of `GPU/base_cuda_sampler.py`'s `prepare_self_feeding` /
//! `upload_slot` / `download_slot` / `launch_self_feeding` /
//! `_run_streaming_loop`, adapted to the `Sampler::sample_stream` contract
//! (`blocking_recv`/`blocking_send` over bounded tokio mpsc channels) and to
//! `GPU/slot_rotation.py`'s `SlotState` bookkeeping.
//!
//! Deviation from the reference driver: the Python cold start blocks for
//! `num_k` models unconditionally, which assumes an effectively-infinite
//! feeder. Jobs here arrive credit-gated from a coordinator that may send
//! fewer than `stream_width()` jobs in total (e.g. a short drive run), so an
//! unconditional blocking cold start could hang forever. This driver instead
//! blocks for the first job, then drains whatever else is already queued
//! (bounded wait), and launches with however many nonces that filled — still
//! correct, just not guaranteed to hit full width on a very short run.

use crate::cuda_device::CudaDevice;
use crate::sampler::SampleError;
use crate::topology::{fill_h_j, SelfFeedingTopology};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use quip_miner_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_miner_core::{
    Algorithm, IsingGraph, SampleParams, SamplerResult, StreamJob, StreamResult,
};
use quip_proto::v1::RejectReason;
use quip_protocol::scoring::energy_milli;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{Receiver, Sender};

const CTRL_STRIDE: usize = 8;
const CTRL_EXIT_NOW: usize = 6;
const SLOTS_PER_NONCE: usize = 3;
const SLOT_READY: i32 = 1;
const SLOT_COMPLETE: i32 = 3;

/// Per-algorithm constants dictated by the verbatim kernel's fixed-size
/// arrays / thread-block shape. Not tunable without editing the kernel.
struct AlgoLimits {
    /// CUDA blocks (SMs) launched per nonce.
    sms_per_nonce: usize,
    /// Largest `N` the kernel's fixed-size per-thread/shared state supports.
    max_nodes: usize,
    /// Largest reads-per-nonce this driver allocates for.
    max_reads: usize,
}

fn algo_limits(algorithm: Algorithm) -> AlgoLimits {
    match algorithm {
        // 1 block (1 SM) per nonce; `if (tid < num_reads)` in a 256-thread
        // block hard-caps reads/nonce; `unpacked_state[5000]` caps N.
        Algorithm::Sa => AlgoLimits {
            sms_per_nonce: 1,
            max_nodes: 5000,
            max_reads: 256,
        },
        // `shared_state[4800]` caps N. reads/nonce isn't block-capped (work
        // is chunked across `sms_per_nonce` blocks) but is held to the same
        // 256 for a uniform, generous device-memory bound.
        Algorithm::Gibbs => AlgoLimits {
            sms_per_nonce: 4,
            max_nodes: 4800,
            max_reads: 256,
        },
    }
}

/// Backend-facing read cap for `Sampler::max_reads`. `max_nodes` is instead
/// hardcoded directly on [`crate::CUDA_SA_IDENTITY`] /
/// [`crate::CUDA_GIBBS_IDENTITY`] (kept next to `algo_limits` in spirit —
/// `BackendIdentity` is a `const`, so it can't call a non-`const fn` here).
///
/// # Examples
///
/// ```
/// use quip_miner_cuda::streaming::max_reads;
/// use quip_miner_cuda::Algorithm;
///
/// // Both kernels are held to one 256-thread block's worth of reads.
/// assert_eq!(max_reads(Algorithm::Sa), 256);
/// assert_eq!(max_reads(Algorithm::Gibbs), 256);
/// ```
pub fn max_reads(algorithm: Algorithm) -> u32 {
    algo_limits(algorithm).max_reads as u32
}

fn tile_i32(src: &[i32], times: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(src.len() * times);
    for _ in 0..times {
        out.extend_from_slice(src);
    }
    out
}

/// Geometric beta schedule cast to f32 for kernel upload, plus sweeps-per-beta.
pub(crate) fn build_beta_schedule(
    graph: &IsingGraph,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
) -> (Vec<f32>, usize) {
    let sweeps_per = sweeps_per_beta.max(1);
    let num_betas = (num_sweeps / sweeps_per).max(1);
    let (hot, cold) = beta_range.unwrap_or_else(|| default_ising_beta_range(graph));
    let sched: Vec<f32> = geometric_beta_schedule(hot, cold, num_betas)
        .iter()
        .map(|&b| b as f32)
        .collect();
    (sched, sweeps_per)
}

pub(crate) fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
}

/// Unpack one read's bit-packed spins (LSB-first per byte, bit=1 -> -1,
/// bit=0 -> +1; matches the kernel's `set_spin_packed`).
fn unpack_spins(packed: &[i8], n: usize) -> Vec<i8> {
    let mut spins = vec![1i8; n];
    for (i, s) in spins.iter_mut().enumerate() {
        let byte = packed[i >> 3] as u8;
        let bit = (byte >> (i & 7)) & 1;
        *s = if bit == 1 { -1 } else { 1 };
    }
    spins
}

/// A running self-feeding kernel + its 3-slot rotating buffers.
///
/// Buffers are allocated with the device's event tracking disabled (see
/// `CudaDevice::open`): the persistent kernel on `stream_compute` and the
/// slot upload/download traffic on `stream_transfer` intentionally race at
/// the byte-range level, arbitrated by the kernel's own volatile ctrl
/// protocol, not by stream ordering. `Drop` signals exit and synchronizes
/// `stream_compute` before any buffer is freed, so this is safe as long as
/// no other code independently frees these slices first.
struct SelfFeedingSession<'a> {
    device: &'a CudaDevice,
    algorithm: Algorithm,
    topology: SelfFeedingTopology,
    num_nonces: usize,
    active_nonces: usize,
    reads_per_nonce: usize,
    max_packed_size: usize,

    stream_compute: Arc<CudaStream>,
    stream_transfer: Arc<CudaStream>,

    d_row_ptr: CudaSlice<i32>,
    d_col_ind: CudaSlice<i32>,
    d_j: CudaSlice<i8>,
    d_h: CudaSlice<i8>,
    d_samples: CudaSlice<i8>,
    d_energies: CudaSlice<i32>,
    d_ctrl: CudaSlice<i32>,
    d_beta: CudaSlice<f32>,

    algo_state: AlgoState,

    // Host staging, reused across uploads to avoid a realloc per model.
    stage_j: Vec<i8>,
    stage_h: Vec<i8>,

    launched: bool,
}

/// Algorithm-specific buffers, kept out of `Option`s: which variant is
/// populated always matches `SelfFeedingSession::algorithm` by construction,
/// so `launch()` destructures it directly instead of unwrapping an `Option`
/// known-Some-by-invariant.
enum AlgoState {
    Sa {
        d_delta_energy: CudaSlice<i8>,
    },
    Gibbs {
        d_block_starts: CudaSlice<i32>,
        d_block_counts: CudaSlice<i32>,
        d_color_nodes: CudaSlice<i32>,
        num_colors: i32,
        chunks_per_model: i32,
        reads_per_chunk: i32,
    },
}

impl<'a> SelfFeedingSession<'a> {
    fn build(
        device: &'a CudaDevice,
        algorithm: Algorithm,
        topology: SelfFeedingTopology,
        num_nonces: usize,
        reads_per_nonce: usize,
        max_num_betas: usize,
    ) -> Result<Self, SampleError> {
        let limits = algo_limits(algorithm);
        let n = topology.n;
        // Defense in depth: `CUDA_SA_IDENTITY`/`CUDA_GIBBS_IDENTITY.max_nodes`
        // already reject an oversized job in `job.rs` before it reaches the
        // sampler; this catches any future drift between those consts and
        // the kernel's actual fixed-size array bounds before it becomes a
        // kernel-side buffer overrun instead of a clean error.
        if n > limits.max_nodes {
            return Err(SampleError::GraphTooLarge {
                n,
                limit: limits.max_nodes,
            });
        }
        let nnz_alloc = topology.nnz.max(1);
        let max_packed_size = n.div_ceil(8).max(1);
        let total_slots = num_nonces * SLOTS_PER_NONCE;

        let ctx = &device.ctx;
        let stream_compute = ctx.new_stream()?;
        let stream_transfer = ctx.new_stream()?;

        let row_ptr = if topology.row_ptr.is_empty() {
            vec![0i32]
        } else {
            topology.row_ptr.clone()
        };
        let col_ind = if topology.nnz == 0 {
            vec![0i32]
        } else {
            topology.col_ind.clone()
        };
        let d_row_ptr = stream_compute.clone_htod(&row_ptr)?;
        let d_col_ind = stream_compute.clone_htod(&col_ind)?;

        let d_j = stream_compute.alloc_zeros::<i8>(total_slots * nnz_alloc)?;
        let d_h = stream_compute.alloc_zeros::<i8>(total_slots * n.max(1))?;
        let d_samples =
            stream_compute.alloc_zeros::<i8>(total_slots * reads_per_nonce * max_packed_size)?;
        let d_energies =
            stream_compute.alloc_zeros::<i32>((total_slots * reads_per_nonce).max(1))?;
        let d_ctrl = stream_compute.alloc_zeros::<i32>(num_nonces * CTRL_STRIDE)?;
        let d_beta = stream_compute.alloc_zeros::<f32>(max_num_betas.max(1))?;

        let algo_state = match algorithm {
            Algorithm::Sa => {
                let total_threads = num_nonces * 256;
                AlgoState::Sa {
                    d_delta_energy: stream_compute.alloc_zeros::<i8>(total_threads * n.max(1))?,
                }
            }
            Algorithm::Gibbs => {
                let starts = tile_i32(&topology.colors.starts, num_nonces);
                let counts = tile_i32(&topology.colors.counts, num_nonces);
                let starts = if starts.is_empty() {
                    vec![0i32]
                } else {
                    starts
                };
                let counts = if counts.is_empty() {
                    vec![0i32]
                } else {
                    counts
                };
                let nodes = if topology.colors.nodes.is_empty() {
                    vec![0i32]
                } else {
                    topology.colors.nodes.clone()
                };
                AlgoState::Gibbs {
                    d_block_starts: stream_compute.clone_htod(&starts)?,
                    d_block_counts: stream_compute.clone_htod(&counts)?,
                    d_color_nodes: stream_compute.clone_htod(&nodes)?,
                    num_colors: topology.colors.num_colors,
                    chunks_per_model: limits.sms_per_nonce as i32,
                    reads_per_chunk: reads_per_nonce.div_ceil(limits.sms_per_nonce) as i32,
                }
            }
        };

        // Every device buffer above is zero-initialized by an `alloc_zeros`
        // memset enqueued on `stream_compute`. Per-slot `h`/`J`/ctrl uploads
        // (`upload_slot`) and the beta-schedule upload (`upload_beta_schedule`)
        // run on `stream_transfer`, and this device has event tracking disabled
        // (see `CudaDevice::open`), so the two streams are unordered. A memset
        // that lands *after* its buffer's upload silently re-zeros it — most
        // visibly the tiny, fast beta upload, whose memset sits behind the large
        // `d_samples`/`d_delta_energy` memsets and so lands late, leaving the
        // kernel to anneal against an all-zero beta schedule (no annealing at
        // all -> garbage energies that scale with how long the memset queue runs
        // vs. the upload). Draining `stream_compute` here guarantees all
        // zero-init completes before any upload can race with it. From this
        // point on the intended byte-range racing between the persistent kernel
        // and slot traffic is arbitrated by the volatile ctrl protocol as
        // documented; only this one-time initialization needed ordering.
        stream_compute.synchronize()?;

        Ok(Self {
            device,
            algorithm,
            topology,
            num_nonces,
            active_nonces: 0,
            reads_per_nonce,
            max_packed_size,
            stream_compute,
            stream_transfer,
            d_row_ptr,
            d_col_ind,
            d_j,
            d_h,
            d_samples,
            d_energies,
            d_ctrl,
            d_beta,
            algo_state,
            stage_j: vec![0i8; nnz_alloc],
            stage_h: vec![0i8; n.max(1)],
            launched: false,
        })
    }

    fn upload_beta_schedule(&mut self, sched: &[f32]) -> Result<(), SampleError> {
        if sched.is_empty() {
            return Ok(());
        }
        self.stream_transfer
            .memcpy_htod(sched, &mut self.d_beta.slice_mut(0..sched.len()))?;
        Ok(())
    }

    /// Upload one job's `h`/`J` into `(nonce_id, slot_id)` and mark it READY.
    /// The job's graph must already be known compatible with `self.topology`
    /// (same `N`/edges) — checked by the caller via [`SessionKey`].
    fn upload_slot(
        &mut self,
        nonce_id: usize,
        slot_id: usize,
        graph: &IsingGraph,
    ) -> Result<(), SampleError> {
        let slot_idx = nonce_id * SLOTS_PER_NONCE + slot_id;
        let nnz = self.topology.nnz;
        let n = self.topology.n;

        // `stage_h` is sized from `self.topology.n`, but the `h` produced
        // below is sized from `graph.h`. `SessionKey::matches` guarantees
        // they agree for every job that reaches here, and `sample_one`
        // builds the topology from this very graph — but that guarantee
        // lives in other functions, so restate it here rather than let a
        // future caller turn it into an out-of-bounds `copy_from_slice`.
        if graph.h.len() != n {
            return Err(SampleError::Driver(format!(
                "upload_slot: job graph N={} does not match session topology N={n}",
                graph.h.len()
            )));
        }

        self.stage_j[..nnz.max(1)].fill(0);
        self.stage_h[..n.max(1)].fill(0);
        let (j, h) = fill_h_j(&self.topology, graph);
        self.stage_j[..j.len()].copy_from_slice(&j);
        self.stage_h[..h.len()].copy_from_slice(&h);

        let nnz_alloc = self.topology.nnz.max(1);
        let n_alloc = self.topology.n.max(1);
        let j_start = slot_idx * nnz_alloc;
        let h_start = slot_idx * n_alloc;
        self.stream_transfer.memcpy_htod(
            &self.stage_j[..nnz_alloc],
            &mut self.d_j.slice_mut(j_start..j_start + nnz_alloc),
        )?;
        self.stream_transfer.memcpy_htod(
            &self.stage_h[..n_alloc],
            &mut self.d_h.slice_mut(h_start..h_start + n_alloc),
        )?;

        // Zero this slot's output region so a stale prior model's samples
        // can't leak through if the kernel writes fewer bytes than expected.
        let sample_start = slot_idx * self.reads_per_nonce * self.max_packed_size;
        let sample_len = self.reads_per_nonce * self.max_packed_size;
        let zeros = vec![0i8; sample_len];
        self.stream_transfer.memcpy_htod(
            &zeros,
            &mut self
                .d_samples
                .slice_mut(sample_start..sample_start + sample_len),
        )?;

        let ctrl_offset = nonce_id * CTRL_STRIDE + slot_id;
        self.stream_transfer.memcpy_htod(
            &[SLOT_READY],
            &mut self.d_ctrl.slice_mut(ctrl_offset..ctrl_offset + 1),
        )?;
        Ok(())
    }

    /// Download and unpack one COMPLETE slot's samples into per-read spins.
    fn download_slot(&self, nonce_id: usize, slot_id: usize) -> Result<Vec<Vec<i8>>, SampleError> {
        let slot_idx = nonce_id * SLOTS_PER_NONCE + slot_id;
        let sample_start = slot_idx * self.reads_per_nonce * self.max_packed_size;
        let sample_len = self.reads_per_nonce * self.max_packed_size;
        let packed: Vec<i8> = self.stream_transfer.clone_dtoh(
            &self
                .d_samples
                .slice(sample_start..sample_start + sample_len),
        )?;
        let n = self.topology.n;
        Ok((0..self.reads_per_nonce)
            .map(|r| {
                let start = r * self.max_packed_size;
                unpack_spins(&packed[start..start + self.max_packed_size], n)
            })
            .collect())
    }

    fn poll_ctrl(&self) -> Result<Vec<i32>, SampleError> {
        Ok(self.stream_transfer.clone_dtoh(&self.d_ctrl)?)
    }

    fn launch(
        &mut self,
        active_nonces: usize,
        num_betas: i32,
        sweeps_per_beta: i32,
        seed: u32,
    ) -> Result<(), SampleError> {
        self.active_nonces = active_nonces;
        let limits = algo_limits(self.algorithm);
        let num_blocks = (active_nonces * limits.sms_per_nonce) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = self.topology.n as i32;
        let nnz = self.topology.nnz as i32;
        let max_packed = self.max_packed_size as i32;
        let num_nonces = self.num_nonces as i32;
        let reads_per_nonce = self.reads_per_nonce as i32;

        // Buffer args are pushed as shared refs regardless of which side
        // (host driver vs. persistent kernel) mutates them: event tracking
        // is disabled for this device (see `CudaDevice::open`), so
        // cudarc's read/write distinction is inert here — the kernel's own
        // volatile ctrl protocol is the actual synchronization.
        match &self.algo_state {
            AlgoState::Sa { d_delta_energy } => {
                let mut b = self.stream_compute.launch_builder(&self.device.sa);
                b.arg(&self.d_row_ptr);
                b.arg(&self.d_col_ind);
                b.arg(&self.d_j);
                b.arg(&self.d_h);
                b.arg(&self.d_samples);
                b.arg(&self.d_energies);
                b.arg(&self.d_beta);
                b.arg(&num_betas);
                b.arg(&sweeps_per_beta);
                b.arg(&self.d_ctrl);
                b.arg(&num_nonces);
                b.arg(&reads_per_nonce);
                b.arg(&n);
                b.arg(&nnz);
                b.arg(&max_packed);
                b.arg(&seed);
                b.arg(d_delta_energy);
                b.arg(&n);
                // SAFETY: `launch_builder` pushes arguments positionally with
                // no compile-time check against the device-side signature, so
                // the obligation is that the 18 `b.arg` calls above mirror
                // `cuda_sa_self_feeding` in `kernels/sa.cu` in order, type and
                // count — they do, ending in the `delta_energy_workspace` /
                // `max_N` pair. Every device buffer the kernel indexes is
                // bounded by scalars it also receives: `n`, `nnz`,
                // `max_packed`, `num_nonces` and `reads_per_nonce` are the
                // exact values `build` sized `d_h`/`d_j`/`d_samples`/
                // `d_energies`/`d_ctrl`/`d_delta_energy` from, and `build`
                // rejected `n > limits.max_nodes` (see the `GraphTooLarge`
                // guard), which is what keeps the kernel's fixed-size
                // `unpacked_state[5000]` in range. The buffers outlive the
                // launch: `Drop` signals exit and synchronizes
                // `stream_compute` before any `CudaSlice` field is freed.
                unsafe { b.launch(cfg) }?;
            }
            AlgoState::Gibbs {
                d_block_starts,
                d_block_counts,
                d_color_nodes,
                num_colors,
                chunks_per_model,
                reads_per_chunk,
            } => {
                let mut b = self.stream_compute.launch_builder(&self.device.gibbs);
                b.arg(&self.d_row_ptr);
                b.arg(&self.d_col_ind);
                b.arg(d_block_starts);
                b.arg(d_block_counts);
                b.arg(d_color_nodes);
                b.arg(num_colors);
                b.arg(&self.d_beta);
                b.arg(&num_betas);
                b.arg(&sweeps_per_beta);
                b.arg(&self.d_j);
                b.arg(&self.d_h);
                b.arg(&self.d_samples);
                b.arg(&self.d_energies);
                b.arg(&self.d_ctrl);
                b.arg(&num_nonces);
                let sms_per_nonce = limits.sms_per_nonce as i32;
                b.arg(&sms_per_nonce);
                b.arg(&reads_per_nonce);
                b.arg(&n);
                b.arg(&nnz);
                b.arg(&max_packed);
                b.arg(chunks_per_model);
                b.arg(reads_per_chunk);
                b.arg(&seed);
                let update_mode = 0i32; // heat-bath Gibbs (no metropolis knob on this wire path)
                b.arg(&update_mode);
                // SAFETY: same obligation as the SA arm above. The 24 `b.arg`
                // calls mirror `cuda_gibbs_self_feeding` in `kernels/gibbs.cu`
                // in order, type and count, ending in `base_seed` /
                // `update_mode`. `n`, `nnz`, `max_packed`, `num_nonces` and
                // `reads_per_nonce` are the values `build` sized the slot
                // buffers from; `num_colors`, `chunks_per_model` and
                // `reads_per_chunk` likewise bound the color-block arrays
                // `build` tiled per nonce. `build` rejected
                // `n > limits.max_nodes`, keeping the kernel's fixed-size
                // `shared_state[4800]` in range. The buffers outlive the
                // launch: `Drop` signals exit and synchronizes
                // `stream_compute` before any `CudaSlice` field is freed.
                unsafe { b.launch(cfg) }?;
            }
        }
        self.launched = true;
        Ok(())
    }

    fn signal_exit(&mut self) -> Result<(), SampleError> {
        if !self.launched {
            return Ok(());
        }
        let exit = vec![1i32; 1];
        for nonce_id in 0..self.active_nonces {
            let off = nonce_id * CTRL_STRIDE + CTRL_EXIT_NOW;
            self.stream_transfer
                .memcpy_htod(&exit, &mut self.d_ctrl.slice_mut(off..off + 1))?;
        }
        Ok(())
    }

    fn wait_exit(&mut self) -> Result<(), SampleError> {
        if self.launched {
            self.stream_compute.synchronize()?;
            self.launched = false;
        }
        Ok(())
    }
}

impl Drop for SelfFeedingSession<'_> {
    fn drop(&mut self) {
        // Kernel must genuinely exit before any CudaSlice field is freed:
        // event tracking is disabled for this device (see CudaDevice::open),
        // so there is no automatic wait built into the Drop of those fields.
        //
        // That makes a failed synchronize unrecoverable rather than
        // ignorable. If we return without establishing that the kernel has
        // stopped, the `CudaSlice` fields are freed immediately afterwards
        // and a still-running persistent kernel keeps writing into
        // `d_samples`/`d_ctrl`/`d_energies` — a use-after-free on the device,
        // the exact scenario `CudaDevice::open`'s safety note rules out.
        // Leaking device memory is sound and aborting is sound; freeing
        // memory that is still in use is not. So escalate instead of giving
        // up: drain the compute stream, then the whole context, then abort.
        drop(self.signal_exit());
        let Err(e) = self.wait_exit() else {
            return;
        };
        eprintln!("cuda self-feeding teardown: stream sync failed: {e}");
        // A whole-context synchronize also covers work the per-stream sync
        // could not observe, so it is a genuine second chance rather than a
        // retry of the same call.
        if let Err(e) = self.device.ctx.synchronize() {
            eprintln!(
                "cuda self-feeding teardown: context sync also failed: {e}; aborting rather than \
                 freeing device buffers the persistent kernel may still be writing"
            );
            std::process::abort();
        }
    }
}

/// Structural + sampling-config identity a self-feeding session is built
/// for. A job can reuse the running session iff it matches: same graph
/// (topology, so the CSR/coloring/edge positions stay valid) and same
/// beta-schedule shape (so the shared beta buffer stays valid). `num_reads`
/// only needs to fit under the session's established per-slot capacity.
#[derive(Clone)]
struct SessionKey {
    n: usize,
    edges: Vec<(usize, usize)>,
    reads_per_nonce: usize,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
}

impl SessionKey {
    fn seed(job: &StreamJob, reads_per_nonce: usize) -> Self {
        Self {
            n: job.graph.h.len(),
            edges: job.graph.edges.clone(),
            reads_per_nonce,
            num_sweeps: job.params.num_sweeps,
            sweeps_per_beta: job.params.sweeps_per_beta.max(1),
            beta_range: job.params.beta_range,
        }
    }

    fn matches(&self, job: &StreamJob) -> bool {
        self.n == job.graph.h.len()
            && self.edges == job.graph.edges
            && self.num_sweeps == job.params.num_sweeps
            && self.sweeps_per_beta == job.params.sweeps_per_beta.max(1)
            && self.beta_range == job.params.beta_range
            && job.params.num_reads.max(1) <= self.reads_per_nonce
    }
}

/// Run one job through a dedicated single-nonce self-feeding session:
/// upload to slot 0, launch, poll to completion, download, tear down.
/// Used by [`crate::sampler::sample_ising`] (the `Sampler::sample` path)
/// and as the streaming loop's oversized/incompatible-job fallback.
///
/// A graph with no nodes is answered directly with empty reads and never
/// touches the device.
///
/// # Errors
///
/// - [`SampleError::GraphTooLarge`] if `graph` has more nodes than the
///   chosen kernel's fixed-size per-thread/shared state supports. This is
///   permanent for this backend — the limit is compiled into the kernel, so
///   the caller should reject the job rather than retry it.
/// - [`SampleError::KernelTimeout`] if the kernel has not marked slot 0
///   COMPLETE within 120 seconds of launch.
/// - [`SampleError::Cuda`] or [`SampleError::Driver`] for a CUDA driver
///   fault at any stage: creating the session's streams and device buffers,
///   uploading the beta schedule or the job's `h`/`J`, launching the kernel,
///   polling the ctrl mailbox, or downloading the packed samples.
///
/// # Examples
///
/// ```no_run
/// use quip_miner_cuda::cuda_device::CudaDevice;
/// use quip_miner_cuda::streaming::sample_one;
/// use quip_miner_cuda::{Algorithm, IsingGraph, SampleParams};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = CudaDevice::open(0)?;
/// let graph = IsingGraph::new(vec![1.0, -1.0], vec![1.0], vec![(0, 1)]);
/// let params = SampleParams {
///     num_reads: 8,
///     ..SampleParams::default()
/// };
/// let reads = sample_one(&device, &graph, &params, Algorithm::Sa)?;
/// let best = reads.iter().map(|r| r.energy_milli).min();
/// # let _ = best;
/// # Ok(())
/// # }
/// ```
pub fn sample_one(
    device: &CudaDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    algorithm: Algorithm,
) -> Result<Vec<SamplerResult>, SampleError> {
    let n = graph.num_nodes();
    if n == 0 {
        let reads = params.num_reads.max(1);
        return Ok((0..reads)
            .map(|_| SamplerResult {
                spins: vec![],
                energy_milli: 0,
            })
            .collect());
    }

    let limits = algo_limits(algorithm);
    let reads_per_nonce = params.num_reads.max(1).min(limits.max_reads);
    let (beta, sweeps_per_beta) = build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );
    let num_betas = beta.len() as i32;
    let topology = SelfFeedingTopology::build(graph);

    let mut sess =
        SelfFeedingSession::build(device, algorithm, topology, 1, reads_per_nonce, beta.len())?;
    sess.upload_beta_schedule(&beta)?;
    sess.upload_slot(0, 0, graph)?;
    let seed = (params.seed as u32).wrapping_add(1);
    sess.launch(1, num_betas, sweeps_per_beta as i32, seed)?;

    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let ctrl = sess.poll_ctrl()?;
        if ctrl[0] == SLOT_COMPLETE {
            break;
        }
        if Instant::now() > deadline {
            return Err(SampleError::KernelTimeout);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let reads = sess.download_slot(0, 0)?;
    sess.signal_exit()?;
    sess.wait_exit()?;

    Ok(reads
        .into_iter()
        .take(params.num_reads.max(1))
        .map(|spins| score_spins(&spins, graph))
        .collect())
}

/// The job a nonce is currently annealing, the device slot holding it, and
/// when it was handed to the kernel (for `device_access_time_us`).
struct ActiveSlot {
    slot: u8,
    job: StreamJob,
    started: Instant,
}

/// Per-nonce ACTIVE/NEXT slot bookkeeping. Port of `GPU/slot_rotation.py`.
///
/// Occupancy is encoded exactly once, by the `Option`s: an idle nonce has a
/// single representation (`None`), and a slot index only exists where a job
/// exists. There is no `-1` sentinel to cast into a buffer index, so
/// "occupied" and "which slot" cannot disagree.
#[derive(Default)]
struct SlotState {
    active: Option<ActiveSlot>,
    next: Option<(u8, StreamJob)>,
}

impl SlotState {
    fn is_idle(&self) -> bool {
        self.active.is_none()
    }

    fn needs_next(&self) -> bool {
        self.active.is_some() && self.next.is_none()
    }

    /// A slot index this nonce is not already using, or `None` if ACTIVE and
    /// NEXT between them account for every slot.
    fn free_slot(&self) -> Option<u8> {
        let active = self.active.as_ref().map(|a| a.slot);
        let next = self.next.as_ref().map(|(slot, _)| *slot);
        (0..SLOTS_PER_NONCE as u8).find(|i| Some(*i) != active && Some(*i) != next)
    }

    fn assign_active(&mut self, slot: u8, job: StreamJob) {
        self.active = Some(ActiveSlot {
            slot,
            job,
            started: Instant::now(),
        });
    }

    fn assign_next(&mut self, slot: u8, job: StreamJob) {
        self.next = Some((slot, job));
    }

    /// Retire the ACTIVE job and promote NEXT (if any) into its place.
    /// Returns the retired job, or `None` if this nonce was already idle.
    fn rotate_on_completion(&mut self) -> Option<ActiveSlot> {
        let done = self.active.take()?;
        self.active = self.next.take().map(|(slot, job)| ActiveSlot {
            slot,
            job,
            started: Instant::now(),
        });
        Some(done)
    }

    /// Take every job this nonce still holds (ACTIVE then NEXT), leaving it
    /// idle. Used to reject in-flight work when a session aborts, so no job
    /// is dropped without a `StreamResult`.
    fn drain_jobs(&mut self) -> Vec<StreamJob> {
        let mut jobs = Vec::new();
        if let Some(active) = self.active.take() {
            jobs.push(active.job);
        }
        if let Some((_, job)) = self.next.take() {
            jobs.push(job);
        }
        jobs
    }
}

/// `Sampler::stream_width()` for the CUDA backend: `max_sms / sms_per_nonce`.
///
/// # Examples
///
/// ```no_run
/// use quip_miner_cuda::cuda_device::CudaDevice;
/// use quip_miner_cuda::streaming::stream_width;
/// use quip_miner_cuda::Algorithm;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = CudaDevice::open(0)?;
/// // Gibbs spends 4 SMs per nonce, so it runs a quarter of SA's width.
/// assert!(stream_width(&device, Algorithm::Sa) >= stream_width(&device, Algorithm::Gibbs));
/// # Ok(())
/// # }
/// ```
pub fn stream_width(device: &CudaDevice, algorithm: Algorithm) -> usize {
    (device.max_sms / algo_limits(algorithm).sms_per_nonce).max(1)
}

enum Pull {
    Job(StreamJob),
    Mismatch(StreamJob),
    Empty,
    Closed,
}

fn try_pull(jobs: &mut Receiver<StreamJob>, key: &SessionKey) -> Pull {
    match jobs.try_recv() {
        Ok(j) if key.matches(&j) => Pull::Job(j),
        Ok(j) => Pull::Mismatch(j),
        Err(TryRecvError::Empty) => Pull::Empty,
        Err(TryRecvError::Disconnected) => Pull::Closed,
    }
}

fn send_reject(out: &Sender<StreamResult>, job: StreamJob, reason: RejectReason) {
    // Discarded deliberately: `blocking_send` only fails once the result
    // channel is closed, i.e. the consumer is gone and no result of any kind
    // can still be delivered. Every caller either follows this with a send of
    // its own (see `emit_completion`, which does set `exhausted` on that same
    // condition) or is already on an abort path, so there is nothing a
    // rejection could recover here.
    drop(out.blocking_send(StreamResult {
        job_id: job.job_id,
        result: Err(reason),
        device_access_time_us: 0,
    }));
}

/// Score one completed job's downloaded reads and emit its `StreamResult`.
///
/// A failed download is reported as `Overloaded` rather than banked as an
/// empty success: telling the coordinator a job succeeded with zero reads
/// loses the work permanently, whereas a rejection lets it retry.
///
/// Returns `false` once the result channel has closed.
fn emit_completion(
    out: &Sender<StreamResult>,
    done: ActiveSlot,
    reads: Result<Vec<Vec<i8>>, SampleError>,
) -> bool {
    let device_access_time_us = done.started.elapsed().as_micros() as u64;
    let reads = match reads {
        Ok(reads) => reads,
        Err(e) => {
            eprintln!("cuda streaming slot download failed: {e}");
            send_reject(out, done.job, RejectReason::Overloaded);
            return true;
        }
    };
    let results: Vec<SamplerResult> = reads
        .into_iter()
        .take(done.job.params.num_reads.max(1))
        .map(|spins| score_spins(&spins, &done.job.graph))
        .collect();
    out.blocking_send(StreamResult {
        job_id: done.job.job_id,
        result: Ok(results),
        device_access_time_us,
    })
    .is_ok()
}

/// Drive the self-feeding streaming loop for the lifetime of `jobs`: keep up
/// to [`stream_width`] models in flight across one (or, on a topology/param
/// change, successive) persistent kernel launches, emitting results in
/// completion order.
///
/// Returns once `jobs` is closed and every job it delivered has produced a
/// [`StreamResult`] — success, or a `RejectReason` if the device failed.
///
/// # Examples
///
/// ```no_run
/// use quip_miner_core::{StreamJob, StreamResult};
/// use quip_miner_cuda::cuda_device::CudaDevice;
/// use quip_miner_cuda::streaming::run_stream;
/// use quip_miner_cuda::Algorithm;
/// use tokio::sync::mpsc::channel;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = CudaDevice::open(0)?;
/// let (job_tx, job_rx) = channel::<StreamJob>(16);
/// let (res_tx, _res_rx) = channel::<StreamResult>(16);
///
/// // A real caller feeds `job_tx` from another task; closing it is what
/// // eventually lets `run_stream` return.
/// drop(job_tx);
/// run_stream(&device, Algorithm::Sa, job_rx, res_tx);
/// # Ok(())
/// # }
/// ```
// OWN-4: `out` is taken by value deliberately. Dropping the `Sender` when this function
// returns is what closes the results channel and signals stream termination to the coordinator;
// taking `&Sender` would move that drop point to the caller and weaken the contract. The
// signature also mirrors the `Sampler::sample_stream` trait method it implements.
#[allow(clippy::needless_pass_by_value)]
pub fn run_stream(
    device: &CudaDevice,
    algorithm: Algorithm,
    mut jobs: Receiver<StreamJob>,
    out: Sender<StreamResult>,
) {
    let width = stream_width(device, algorithm);
    let limits = algo_limits(algorithm);
    let mut pending_seed: Option<StreamJob> = jobs.blocking_recv();

    'session: while let Some(seed) = pending_seed.take() {
        if seed.graph.num_nodes() == 0 {
            // Degenerate empty-graph job: no kernel needed, answer directly.
            let reads = seed.params.num_reads.max(1);
            if out
                .blocking_send(StreamResult {
                    job_id: seed.job_id,
                    result: Ok((0..reads)
                        .map(|_| SamplerResult {
                            spins: vec![],
                            energy_milli: 0,
                        })
                        .collect()),
                    device_access_time_us: 0,
                })
                .is_err()
            {
                // Result channel closed: nothing we produce from here on can
                // reach anyone. Same condition the completion path treats as
                // `exhausted`, and here there is no in-flight work to drain.
                return;
            }
            pending_seed = jobs.blocking_recv();
            continue 'session;
        }

        let reads_per_nonce = seed.params.num_reads.max(1).min(limits.max_reads);
        let job_seed = seed.params.seed;
        let key = SessionKey::seed(&seed, reads_per_nonce);
        let (beta, sweeps_per_beta) = build_beta_schedule(
            &seed.graph,
            seed.params.num_sweeps,
            seed.params.sweeps_per_beta,
            seed.params.beta_range,
        );
        let num_betas = beta.len() as i32;
        let topology = SelfFeedingTopology::build(&seed.graph);

        let mut sess = match SelfFeedingSession::build(
            device,
            algorithm,
            topology,
            width,
            reads_per_nonce,
            beta.len(),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cuda streaming session build failed: {e}");
                send_reject(&out, seed, RejectReason::Overloaded);
                pending_seed = jobs.blocking_recv();
                continue 'session;
            }
        };
        if sess.upload_beta_schedule(&beta).is_err() {
            send_reject(&out, seed, RejectReason::Overloaded);
            pending_seed = jobs.blocking_recv();
            continue 'session;
        }

        let mut slots: Vec<SlotState> = (0..width).map(|_| SlotState::default()).collect();

        // Cold start: the first job is already in hand; drain whatever else
        // shows up so the launch starts with as much concurrency as
        // possible. `launch_self_feeding`'s grid size is fixed for the
        // kernel's whole lifetime (no adding nonces after launch), so it's
        // worth waiting past the first quiet moment: this keeps pulling
        // until either `width` is reached, a hard cap elapses, or a full
        // `idle_timeout` passes with nothing new arriving (a burst source
        // like a coordinator dispatching its whole staged queue arrives in
        // well under `idle_timeout`; a genuinely slow/empty source gives up
        // after it).
        let mut cold: Vec<StreamJob> = vec![seed];
        let mut mismatch: Option<StreamJob> = None;
        let mut closed = false;
        let cold_hard_cap = Instant::now() + Duration::from_secs(3);
        let idle_timeout = Duration::from_millis(150);
        let mut last_arrival = Instant::now();
        while cold.len() < width && Instant::now() < cold_hard_cap {
            match try_pull(&mut jobs, &key) {
                Pull::Job(j) => {
                    cold.push(j);
                    last_arrival = Instant::now();
                }
                Pull::Mismatch(j) => {
                    mismatch = Some(j);
                    break;
                }
                Pull::Empty => {
                    if last_arrival.elapsed() > idle_timeout {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                Pull::Closed => {
                    closed = true;
                    break;
                }
            }
        }

        let active_nonces = cold.len();
        eprintln!(
            "quip-miner-cuda: self-feeding session launching with {active_nonces}/{width} nonces active"
        );
        for (nonce_id, job) in cold.into_iter().enumerate() {
            if sess.upload_slot(nonce_id, 0, &job.graph).is_err() {
                send_reject(&out, job, RejectReason::Overloaded);
                continue;
            }
            slots[nonce_id].assign_active(0, job);
        }
        // The kernel's RNG seed must actually differ per session, or every
        // session anneals the same trajectories. Wall-clock nanoseconds are
        // the entropy (`Instant::now().elapsed()` is not — it measures the
        // few nanoseconds between its own two calls), mixed with the seed
        // job's own `params.seed` so two sessions starting in the same clock
        // tick still diverge. Multiply-and-take-the-high-word spreads both
        // inputs across the whole u32 rather than exposing raw low bits.
        let wall_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;
        let seed_val = ((wall_nanos ^ job_seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32;
        if sess
            .launch(active_nonces, num_betas, sweeps_per_beta as i32, seed_val)
            .is_err()
        {
            // The cold-start jobs are held in `slots`, not handed off: with
            // no kernel running none of them can ever complete, so reject
            // them here rather than drop them and leave the coordinator
            // waiting on results that will never come.
            for slot in slots.iter_mut().take(active_nonces) {
                for job in slot.drain_jobs() {
                    send_reject(&out, job, RejectReason::Overloaded);
                }
            }
            // A cold-start mismatch is a real job too: carry it to the next
            // session instead of dropping it with the failed launch.
            pending_seed =
                mismatch
                    .take()
                    .or_else(|| if closed { None } else { jobs.blocking_recv() });
            continue 'session;
        }

        let mut exhausted = closed || mismatch.is_some();

        loop {
            // Fill NEXT / revive idle nonces (non-blocking) before polling.
            if !exhausted {
                'nonces: for (nonce_id, slot) in slots.iter_mut().enumerate().take(active_nonces) {
                    while slot.is_idle() || slot.needs_next() {
                        let Some(free) = slot.free_slot() else {
                            break;
                        };
                        match try_pull(&mut jobs, &key) {
                            Pull::Job(j) => {
                                if sess
                                    .upload_slot(nonce_id, usize::from(free), &j.graph)
                                    .is_err()
                                {
                                    send_reject(&out, j, RejectReason::Overloaded);
                                    continue;
                                }
                                if slot.is_idle() {
                                    slot.assign_active(free, j);
                                } else {
                                    slot.assign_next(free, j);
                                }
                            }
                            Pull::Mismatch(j) => {
                                mismatch = Some(j);
                                exhausted = true;
                                break 'nonces;
                            }
                            Pull::Empty => break,
                            Pull::Closed => {
                                exhausted = true;
                                break 'nonces;
                            }
                        }
                    }
                }
            }

            if exhausted
                && slots[..active_nonces]
                    .iter()
                    .all(|s| s.is_idle() && s.next.is_none())
            {
                break;
            }

            let ctrl = match sess.poll_ctrl() {
                Ok(ctrl) => ctrl,
                Err(e) => {
                    // The ctrl mailbox is what tells us a slot finished, so
                    // once it is unreadable no in-flight job can ever be
                    // observed as COMPLETE. Reject everything still held
                    // instead of breaking out and dropping it silently — the
                    // coordinator is blocked on a StreamResult per job.
                    eprintln!("cuda streaming ctrl poll failed: {e}");
                    for slot in slots.iter_mut().take(active_nonces) {
                        for job in slot.drain_jobs() {
                            send_reject(&out, job, RejectReason::Overloaded);
                        }
                    }
                    break;
                }
            };
            let mut found = false;
            for (nonce_id, slot) in slots.iter_mut().enumerate().take(active_nonces) {
                // `active` carries the slot index only when a job occupies
                // it, so there is no idle sentinel to cast and `idx` cannot
                // wrap into another nonce's ctrl word.
                let Some(active) = slot.active.as_ref() else {
                    continue;
                };
                let active_slot = usize::from(active.slot);
                let idx = nonce_id * CTRL_STRIDE + active_slot;
                if ctrl[idx] != SLOT_COMPLETE {
                    continue;
                }
                found = true;
                let reads = sess.download_slot(nonce_id, active_slot);
                // Always `Some`: `active` was matched immediately above.
                let Some(done) = slot.rotate_on_completion() else {
                    continue;
                };
                if !emit_completion(&out, done, reads) {
                    exhausted = true;
                }
            }
            if !found {
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        drop(sess); // signals exit + synchronizes stream_compute (Drop impl)

        pending_seed = mismatch.or_else(|| if closed { None } else { jobs.blocking_recv() });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4-node ring, unit couplings — small enough to reason about by hand.
    fn graph() -> IsingGraph {
        IsingGraph::new(
            vec![1.0, -1.0, 0.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        )
    }

    fn job(id: u8) -> StreamJob {
        StreamJob {
            job_id: vec![id],
            graph: graph(),
            params: SampleParams::default(),
        }
    }

    #[test]
    fn unpack_spins_reads_bits_lsb_first() {
        // 0b0000_0101: bits 0 and 2 set -> those spins are -1, rest +1.
        let spins = unpack_spins(&[0b0000_0101], 8);
        assert_eq!(spins, vec![-1, 1, -1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn unpack_spins_reads_the_sign_bit_as_a_bit() {
        // 0x80 is -128 as i8; the cast back to u8 must not lose bit 7.
        let spins = unpack_spins(&[-128i8], 8);
        assert_eq!(spins, vec![1, 1, 1, 1, 1, 1, 1, -1]);
    }

    #[test]
    fn unpack_spins_spans_bytes_and_stops_at_n() {
        // n=9 reaches into the second byte for exactly one bit.
        let spins = unpack_spins(&[0b0000_0000, 0b0000_0001], 9);
        assert_eq!(spins, vec![1, 1, 1, 1, 1, 1, 1, 1, -1]);
    }

    #[test]
    fn slot_state_starts_idle() {
        let slot = SlotState::default();
        assert!(slot.is_idle());
        assert!(!slot.needs_next());
        assert_eq!(slot.free_slot(), Some(0));
    }

    #[test]
    fn slot_state_rotates_without_active_and_next_colliding() {
        let mut slot = SlotState::default();

        slot.assign_active(0, job(1));
        assert!(!slot.is_idle());
        assert!(
            slot.needs_next(),
            "an occupied nonce with no NEXT wants one"
        );
        assert_eq!(slot.free_slot(), Some(1));

        slot.assign_next(1, job(2));
        assert!(!slot.needs_next());
        // ACTIVE and NEXT hold two of three slots, so the free one is the third.
        assert_eq!(slot.free_slot(), Some(2));
        let active = slot.active.as_ref().expect("active");
        let (next_slot, _) = slot.next.as_ref().expect("next");
        assert_ne!(active.slot, *next_slot, "ACTIVE and NEXT must not collide");

        // Completing ACTIVE hands back job 1 and promotes job 2 in place.
        let done = slot.rotate_on_completion().expect("a job was active");
        assert_eq!(done.slot, 0);
        assert_eq!(done.job.job_id, vec![1]);
        assert!(!slot.is_idle());
        assert_eq!(slot.active.as_ref().map(|a| a.slot), Some(1));
        assert!(slot.next.is_none());
        assert_eq!(slot.free_slot(), Some(0));

        // Completing again with no NEXT queued leaves the nonce idle.
        let done = slot.rotate_on_completion().expect("job 2 was promoted");
        assert_eq!(done.job.job_id, vec![2]);
        assert!(slot.is_idle());
        assert!(slot.rotate_on_completion().is_none());
    }

    #[test]
    fn slot_state_free_slot_avoids_both_held_slots() {
        // ACTIVE and NEXT hold at most two slots, so with three per nonce the
        // remaining one is always offered — whichever two are in use.
        for (active, next, free) in [(0u8, 1u8, 2u8), (2, 0, 1), (1, 2, 0)] {
            let mut slot = SlotState::default();
            slot.assign_active(active, job(1));
            slot.assign_next(next, job(2));
            assert_eq!(slot.free_slot(), Some(free));
        }
    }

    #[test]
    fn slot_state_drain_jobs_yields_active_then_next() {
        let mut slot = SlotState::default();
        slot.assign_active(0, job(1));
        slot.assign_next(1, job(2));

        let drained = slot.drain_jobs();
        assert_eq!(
            drained.iter().map(|j| j.job_id.clone()).collect::<Vec<_>>(),
            vec![vec![1], vec![2]]
        );
        assert!(slot.is_idle());
        assert!(slot.next.is_none());
        assert!(slot.drain_jobs().is_empty());
    }

    #[test]
    fn session_key_matches_an_identical_job() {
        let seed = job(1);
        let key = SessionKey::seed(&seed, 8);
        assert!(key.matches(&seed));
    }

    #[test]
    fn session_key_rejects_a_different_node_count() {
        let key = SessionKey::seed(&job(1), 8);
        let mut other = job(2);
        other.graph.h.push(0.5);
        assert!(!key.matches(&other));
    }

    #[test]
    fn session_key_rejects_different_edges() {
        let key = SessionKey::seed(&job(1), 8);
        let mut other = job(2);
        other.graph.edges = vec![(0, 1), (1, 2), (2, 3), (0, 2)];
        assert!(!key.matches(&other));
    }

    #[test]
    fn session_key_rejects_a_different_sweep_count() {
        let key = SessionKey::seed(&job(1), 8);
        let mut other = job(2);
        other.params.num_sweeps += 1;
        assert!(!key.matches(&other));
    }

    #[test]
    fn session_key_rejects_a_different_sweeps_per_beta() {
        let key = SessionKey::seed(&job(1), 8);
        let mut other = job(2);
        other.params.sweeps_per_beta = 4;
        assert!(!key.matches(&other));
    }

    #[test]
    fn session_key_rejects_a_different_beta_range() {
        let key = SessionKey::seed(&job(1), 8);
        let mut other = job(2);
        other.params.beta_range = Some((0.1, 10.0));
        assert!(!key.matches(&other));
    }

    #[test]
    fn session_key_rejects_more_reads_than_the_session_allocated() {
        let key = SessionKey::seed(&job(1), 8);
        let mut other = job(2);
        other.params.num_reads = 8;
        assert!(key.matches(&other), "exactly the capacity still fits");
        other.params.num_reads = 9;
        assert!(!key.matches(&other));
    }

    #[test]
    fn session_key_treats_zero_and_one_sweeps_per_beta_alike() {
        // Both sides apply `.max(1)`, so 0 and 1 are the same session.
        let mut seed = job(1);
        seed.params.sweeps_per_beta = 0;
        let key = SessionKey::seed(&seed, 8);
        let mut other = job(2);
        other.params.sweeps_per_beta = 1;
        assert!(key.matches(&other));
    }

    #[test]
    fn beta_schedule_length_is_sweeps_over_sweeps_per_beta() {
        let (sched, sweeps_per) = build_beta_schedule(&graph(), 100, 10, Some((0.1, 10.0)));
        assert_eq!(sweeps_per, 10);
        assert_eq!(sched.len(), 10);
    }

    #[test]
    fn beta_schedule_floors_sweeps_per_beta_at_one() {
        // A zero would divide by zero; the floor turns it into one beta per sweep.
        let (sched, sweeps_per) = build_beta_schedule(&graph(), 64, 0, Some((0.1, 10.0)));
        assert_eq!(sweeps_per, 1);
        assert_eq!(sched.len(), 64);
    }

    #[test]
    fn beta_schedule_always_has_at_least_one_beta() {
        // Zero sweeps, or fewer sweeps than sweeps_per_beta, must not yield an
        // empty schedule — the kernel would have nothing to anneal against.
        let (sched, _) = build_beta_schedule(&graph(), 0, 1, Some((0.1, 10.0)));
        assert_eq!(sched.len(), 1);
        let (sched, _) = build_beta_schedule(&graph(), 4, 64, Some((0.1, 10.0)));
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn beta_schedule_runs_hot_to_cold() {
        let (sched, _) = build_beta_schedule(&graph(), 40, 10, Some((0.1, 10.0)));
        assert_eq!(sched.len(), 4);
        // Beta rises as temperature falls, so the schedule is increasing.
        for pair in sched.windows(2) {
            assert!(pair[1] > pair[0], "beta schedule must cool monotonically");
        }
    }

    #[test]
    fn tile_i32_repeats_the_whole_slice() {
        assert_eq!(tile_i32(&[1, 2, 3], 2), vec![1, 2, 3, 1, 2, 3]);
        assert_eq!(tile_i32(&[1, 2, 3], 1), vec![1, 2, 3]);
        assert!(tile_i32(&[1, 2, 3], 0).is_empty());
        assert!(tile_i32(&[], 4).is_empty());
    }

    #[test]
    fn algo_limits_match_the_kernels_fixed_size_arrays() {
        // SA: one block per nonce, `unpacked_state[5000]`.
        let sa = algo_limits(Algorithm::Sa);
        assert_eq!(sa.sms_per_nonce, 1);
        assert_eq!(sa.max_nodes, 5000);
        assert_eq!(sa.max_reads, 256);

        // Gibbs: four blocks per nonce, `shared_state[4800]`.
        let gibbs = algo_limits(Algorithm::Gibbs);
        assert_eq!(gibbs.sms_per_nonce, 4);
        assert_eq!(gibbs.max_nodes, 4800);
        assert_eq!(gibbs.max_reads, 256);
    }

    #[test]
    fn max_reads_reports_the_per_algorithm_cap() {
        assert_eq!(
            max_reads(Algorithm::Sa),
            algo_limits(Algorithm::Sa).max_reads as u32
        );
        assert_eq!(
            max_reads(Algorithm::Gibbs),
            algo_limits(Algorithm::Gibbs).max_reads as u32
        );
    }

    #[test]
    fn score_spins_uses_consensus_energy() {
        let g = graph();
        let spins = vec![1i8, 1, 1, 1];
        let scored = score_spins(&spins, &g);
        assert_eq!(scored.spins, spins);
        assert_eq!(
            scored.energy_milli,
            energy_milli(&spins, &g.h, &g.j, &g.edges)
        );
    }
}

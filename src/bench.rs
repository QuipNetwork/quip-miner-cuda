//! `bench` subcommand: isolated single-model timing + unified JSON emission.
//!
//! Runs the isolated single-shot launch path ([`crate::streaming::bench_many`])
//! over a config grid, folds host-seam and CUDA-event timings into the
//! locked-schema [`crate::schema::BenchRecord`], and (optionally, via `bench
//! fold`) folds an external Nsight report ([`crate::nsight`]) into an
//! already-emitted host JSON.
//!
//! # Running under Nsight
//!
//! `bench run` measures the host seams and the CUDA-event kernel time on its
//! own. For SM efficiency / achieved occupancy and an independent per-launch
//! duration, wrap the process:
//!
//! ```text
//! # Timeline (fast; default for the corpus sweep). One .nsys-rep per run dir.
//! nsys profile --trace=cuda --cuda-memory-usage=false \
//!     -o bench_rep \
//!     quip-cuda-sa bench run --reads 8 --sweeps 1024 --sweeps 8192 \
//!         --sweeps-per-beta 4 --nodes 512 --repeats 5 --out out/
//! nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace \
//!     --format csv --output bench_rep bench_rep.nsys-rep
//! #   -> bench_rep_cuda_gpu_kern_sum.csv, bench_rep_cuda_gpu_trace.csv
//!
//! # Deep-dive metrics (slow; kernel replay). Small slice only.
//! ncu --csv --target-processes all \
//!     --metrics gpu__time_duration.sum,\
//! sm__throughput.avg.pct_of_peak_sustained_elapsed,\
//! sm__warps_active.avg.pct_of_peak_sustained_active,\
//! lts__t_sector_hit_rate.pct,\
//! l1tex__t_sector_hit_rate.pct,\
//! dram__throughput.avg.pct_of_peak_sustained_elapsed \
//!     --kernel-name regex:'cuda_(sa|msa|gibbs)_self_feeding' \
//!     quip-cuda-sa bench run --reads 8 --sweeps 1024 --sweeps-per-beta 4 \
//!         --nodes 512 --repeats 3 --out out/ > ncu_bench.csv
//!
//! # Fold the Nsight report into the per-part JSON the host run emitted:
//! quip-cuda-sa bench fold --host-json out/cuda-sa_n512_s1024.jsonl \
//!     --nsys-kern-sum bench_rep_cuda_gpu_kern_sum.csv \
//!     --sweeps-lo 1024 --sweeps-hi 8192 --kern-sum-hi bench_rep_hi_cuda_gpu_kern_sum.csv \
//!     --out out/cuda-sa_n512_s1024.folded.jsonl
//! ```

use crate::corpus;
use crate::cuda_device::{CudaDevice, MsaLane};
use crate::nsight;
use crate::schema::{BenchRecord, Metric, Part, Scope, Source};
use crate::streaming::{bench_many, bench_plan, stream_width, DeviceTimings};
use crate::{IsingGraph, KernelKind, SampleParams};
use clap::{Args, Subcommand, ValueEnum};
use quip_protocol::chacha8::ChaCha8Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;
use thiserror::Error;
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// `bench` subcommand actions: a timed grid run, or folding an external
/// Nsight report into a run's output.
#[derive(Subcommand, Debug, Clone)]
pub enum BenchAction {
    /// Run the timed bench grid and emit host per-part JSON (+ optional
    /// `tracing-flame` folded stacks).
    Run(RunArgs),
    /// Fold an external `nsys`/`ncu` report into an existing host JSON.
    Fold(FoldArgs),
}

/// MSA path selected for an isolated benchmark.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsaPath {
    /// Plan normally, using the default lane for weighted jobs.
    Auto,
    /// Force the weighted integer lane, including for unit models.
    WeightedI32,
    /// Force the weighted floating-point lane, including for unit models.
    WeightedF32,
}

impl MsaPath {
    fn lane(self) -> Option<MsaLane> {
        match self {
            Self::Auto => None,
            Self::WeightedI32 => Some(MsaLane::I32),
            Self::WeightedF32 => Some(MsaLane::F32),
        }
    }
}

/// Arguments for `bench run`.
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// MSA path (default: auto). Accepted only by the MSA kernel.
    #[arg(long, value_enum)]
    pub msa_path: Option<MsaPath>,
    /// Maximum integer weight. 1 preserves the graph; larger values redraw
    /// fields in -K..=K and coupling magnitudes in 1..=K, preserving signs.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub weights: u32,
    /// Reads per model, at most the kernel's cap (256 for sa and gibbs, 128 for msa).
    #[arg(long, default_value_t = 8)]
    pub reads: u64,
    /// `num_sweeps` to bench at; repeat the flag for a multi-point grid
    /// (e.g. `--sweeps 1024 --sweeps 8192` for the `bench fold` two-point
    /// per-sweep derivation). Defaults to a single 1024-sweep point.
    #[arg(long = "sweeps")]
    pub sweeps: Vec<u64>,
    /// Sweeps per beta rung.
    #[arg(long, default_value_t = 4)]
    pub sweeps_per_beta: u64,
    /// Node count of the synthetic circulant graph (zero fields and unit couplings by default).
    /// Ignored when `--source` is given (the real corpus topology's node
    /// count is used instead). Useful on its own for micro checks.
    #[arg(long, default_value_t = 64)]
    pub nodes: u64,
    /// Neighbours per node of the synthetic circulant graph (even, at least 2).
    /// Node i links to i+1 ..= i+degree/2 (mod n). The default 2 is a ring.
    #[arg(long, default_value_t = 2, conflicts_with = "source")]
    pub degree: u64,
    /// Relabel the synthetic graph's nodes by a seeded random permutation. The
    /// graph is the same; neighbours are no longer adjacent in memory.
    #[arg(long, conflicts_with = "source")]
    pub shuffle: bool,
    /// Models annealed at once, one per nonce, at most the device's stream width.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..))]
    pub nonces: u64,
    /// msa only: hold at most this many spins in shared memory.
    #[arg(long)]
    pub msa_shared_spins: Option<usize>,
    /// Corpus JSONL to bench real models (one `{"nonce":"<64hex>", ...}`
    /// line per model, unknown fields ignored); requires `--topology`.
    /// Redraws each nonce's real `(h, J)` and benches that graph instead of
    /// the synthetic circulant graph.
    #[arg(long, requires = "topology")]
    pub source: Option<PathBuf>,
    /// Topology spec (`{nodes, edges, allowed_h_milli, allowed_j_milli}`)
    /// matching `--source`'s corpus; required together with `--source`.
    #[arg(long, requires = "source")]
    pub topology: Option<PathBuf>,
    /// Cap the number of `--source` corpus records benched.
    #[arg(long)]
    pub limit: Option<usize>,
    /// Measured repeats per grid cell.
    #[arg(long, default_value_t = 5)]
    pub repeats: u64,
    /// Warm-up runs per grid cell, discarded before measurement.
    #[arg(long, default_value_t = 1)]
    pub warmup: u64,
    /// Output directory; one `<backend>_n<N>_s<sweeps>.jsonl` file per cell.
    #[arg(long)]
    pub out: PathBuf,
    /// Optional `tracing-flame` folded-stack output path.
    #[arg(long)]
    pub flame: Option<PathBuf>,
    /// Metadata passthrough for `BenchRecord.nonce` (synthetic problems have
    /// no real nonce). A 64-hex value also seeds `--weights`.
    #[arg(long)]
    pub nonce: Option<String>,
    /// Metadata passthrough for `BenchRecord.topology_hash`.
    #[arg(long = "topology-hash")]
    pub topology_hash: Option<String>,
}

/// Arguments for `bench fold`.
#[derive(Args, Debug, Clone)]
pub struct FoldArgs {
    /// JSONL of [`BenchRecord`]s emitted by `bench run`.
    #[arg(long)]
    pub host_json: PathBuf,
    /// `nsys stats --report cuda_gpu_kern_sum --format csv` output for the
    /// same (or low-sweep) run.
    #[arg(long)]
    pub nsys_kern_sum: PathBuf,
    /// A second `cuda_gpu_kern_sum` CSV from a higher-sweep run, for the
    /// two-point per-sweep derivation. Omit to skip the `sweep` overwrite.
    #[arg(long)]
    pub kern_sum_hi: Option<PathBuf>,
    /// `num_sweeps` used for `--nsys-kern-sum`'s run (the low point).
    #[arg(long)]
    pub sweeps_lo: Option<u64>,
    /// `num_sweeps` used for `--kern-sum-hi`'s run (the high point).
    #[arg(long)]
    pub sweeps_hi: Option<u64>,
    /// Sweeps-per-beta shared by both runs (`S_total = num_betas *
    /// sweeps_per_beta`); needed to convert sweep counts to `S_total`.
    #[arg(long, default_value_t = 1)]
    pub sweeps_per_beta: u64,
    /// Optional `ncu --csv` report for SM efficiency / achieved occupancy.
    #[arg(long)]
    pub ncu_csv: Option<PathBuf>,
    /// Output path for the folded JSONL.
    #[arg(long)]
    pub out: PathBuf,
}

/// Errors from a bench run.
#[derive(Debug, Error)]
pub enum BenchError {
    /// Device open / compile / sampling failure.
    #[error("bench device: {0}")]
    Device(String),
    /// Output file / directory I/O failure.
    #[error("bench io: {0}")]
    Io(String),
    /// Nsight report parse/fold failure (see [`crate::nsight`]).
    #[error("bench fold: {0}")]
    Fold(String),
    /// `--source`/`--topology` corpus load or redraw failure (see [`crate::corpus`]).
    #[error("bench corpus: {0}")]
    Corpus(String),
    /// A `bench run` argument outside what the kernel can execute.
    #[error("bench args: {0}")]
    Args(String),
}

/// One point in the bench config grid.
pub struct CellConfig {
    /// Reads per model.
    pub reads: u64,
    /// `num_sweeps` requested.
    pub num_sweeps: u64,
    /// Sweeps per beta rung, as actually used (post `.max(1)` clamp).
    pub sweeps_per_beta: u64,
    /// `num_betas` the beta schedule was built with; `S_total = num_betas *
    /// sweeps_per_beta`.
    pub num_betas: u64,
    /// Node count `N`.
    pub nodes: u64,
    /// Edge count `E`.
    pub edges: u64,
}

/// Host-seam durations captured for one cell (ns). `jit_ns` is `Some` only on
/// the first run of a process (one-time NVRTC compile), else `None`.
pub struct HostSpans {
    /// `problem_setup` span aggregate.
    pub problem_setup_ns: u64,
    /// `beta_build` span aggregate.
    pub beta_build_ns: u64,
    /// `energy_score` span aggregate (summed over every read).
    pub energy_score_ns: u64,
    /// `jit` span, only on the process's first cell.
    pub jit_ns: Option<u64>,
}

/// Fold one cell's device + host timings into a [`BenchRecord`].
///
/// `dev.kernel_ns` seeds both the device `launch` part (a single call) and
/// the device `sweep` part (`count = num_betas * sweeps_per_beta`, i.e.
/// `S_total`) as a counter-derived estimate; `bench fold` (Task 6,
/// [`crate::nsight::fold_into`]) later overwrites `launch` with the measured
/// `nsys` kernel-sum and `sweep` with the two-point derivation.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn assemble_record(
    backend: &str,
    device: &str,
    nonce: &str,
    topology_hash: &str,
    cfg: &CellConfig,
    dev: &DeviceTimings,
    host: &HostSpans,
    model_total_ns: u64,
) -> BenchRecord {
    let s_total = (cfg.num_betas * cfg.sweeps_per_beta).max(1);
    let mut parts = vec![
        Part::new(
            "problem_setup",
            Scope::Host,
            host.problem_setup_ns,
            1,
            Source::Tracing,
        ),
        Part::new(
            "beta_build",
            Scope::Host,
            host.beta_build_ns,
            1,
            Source::Tracing,
        ),
        Part::new("upload", Scope::Host, dev.upload_ns, 1, Source::CudaEvent),
        Part::new(
            "poll_wait",
            Scope::Host,
            dev.poll_wait_ns,
            1,
            Source::Tracing,
        ),
        Part::new(
            "download",
            Scope::Host,
            dev.download_ns,
            1,
            Source::CudaEvent,
        ),
        Part::new(
            "energy_score",
            Scope::Host,
            host.energy_score_ns,
            cfg.reads,
            Source::Tracing,
        ),
        Part::new("launch", Scope::Device, dev.kernel_ns, 1, Source::CudaEvent),
        Part::new(
            "sweep",
            Scope::Device,
            dev.kernel_ns,
            s_total,
            Source::CounterDerived,
        ),
    ];
    if let Some(jit_ns) = host.jit_ns {
        parts.push(Part::new("jit", Scope::Host, jit_ns, 1, Source::Tracing));
    }
    let mut rec = BenchRecord {
        backend: backend.to_owned(),
        device: device.to_owned(),
        nonce: nonce.to_owned(),
        topology_hash: topology_hash.to_owned(),
        reads: cfg.reads,
        sweeps: cfg.num_sweeps,
        sweeps_per_beta: cfg.sweeps_per_beta,
        nodes: cfg.nodes,
        edges: cfg.edges,
        model_total_ns,
        parts,
        metrics: vec![],
        residual_ns: 0,
    };
    rec.finalize();
    rec
}

/// A deterministic synthetic circulant Ising problem (`h=0`, `j=1`): node i
/// links to i+1 ..= i+degree/2 (mod n). Degree 2 is a ring. With `shuffle`,
/// nodes are relabeled by a `ChaCha8` permutation of that seed, which keeps
/// the graph and destroys index locality.
#[allow(clippy::cast_possible_truncation)] // n fits the supported 64-bit host usize
fn circulant_graph(n: u64, degree: u64, shuffle: Option<[u8; 32]>) -> IsingGraph {
    let n = n as usize;
    let half = (degree / 2) as usize;
    let mut label: Vec<usize> = (0..n).collect();
    if let Some(seed) = shuffle {
        let mut rng = ChaCha8Rng::from_seed(seed);
        for i in (1..n).rev() {
            let k = (rng.next_u32() as usize) % (i + 1);
            label.swap(i, k);
        }
    }
    let mut edges = Vec::with_capacity(n * half);
    for i in 0..n {
        for d in 1..=half {
            edges.push((label[i], label[(i + d) % n.max(1)]));
        }
    }
    let m = edges.len();
    IsingGraph::new(vec![0.0; n], vec![1.0; m], edges)
}

/// Refuse a synthetic `degree` the circulant cannot build on `nodes` nodes,
/// before any edge is allocated: it must be even, at least 2, below `nodes`
/// (else pairs repeat and self-loops appear), and within
/// [`crate::capacity::MSA_MAX_DEGREE`]. SA and Gibbs declare no degree limit
/// of their own, so the msa limit is the bench-wide cap.
fn check_degree(degree: u64, nodes: u64) -> Result<(), BenchError> {
    let max = crate::capacity::MSA_MAX_DEGREE as u64;
    if degree < 2 || !degree.is_multiple_of(2) {
        return Err(BenchError::Args(format!(
            "--degree {degree} must be even and at least 2"
        )));
    }
    if degree >= nodes {
        return Err(BenchError::Args(format!(
            "--degree {degree} must be below --nodes {nodes}"
        )));
    }
    if degree > max {
        return Err(BenchError::Args(format!(
            "--degree {degree} exceeds the kernel limit of {max}"
        )));
    }
    Ok(())
}

/// One graph to bench in the grid: either the synthetic circulant problem or a
/// `--source` corpus record redrawn from its nonce. All models in one `bench
/// run` invocation share a topology (the synthetic circulant graph's `--nodes`, or
/// `--topology`'s node/edge set), so [`run_cell`] can name its output file
/// from the first model alone.
struct BenchModel {
    /// The graph to bench.
    graph: IsingGraph,
    /// Passed through into the emitted [`BenchRecord::nonce`].
    nonce: String,
    /// Passed through into the emitted [`BenchRecord::topology_hash`].
    topology_hash: String,
}

impl BenchModel {
    /// The synthetic circulant model, tagged with `--nonce`/`--topology-hash`
    /// metadata passthrough (synthetic problems have no real identity).
    fn synthetic(args: &RunArgs) -> Self {
        Self {
            graph: circulant_graph(
                args.nodes,
                args.degree,
                args.shuffle
                    .then(|| nonce_seed(args.nonce.as_deref().unwrap_or(""))),
            ),
            nonce: args.nonce.clone().unwrap_or_else(|| "0x0".to_owned()),
            topology_hash: args
                .topology_hash
                .clone()
                .unwrap_or_else(|| "0x0".to_owned()),
        }
    }
}

/// Build the bench grid's models: a `--source` corpus (redrawn per nonce
/// against `--topology`) when given, else the single synthetic circulant problem.
fn load_models(args: &RunArgs) -> Result<Vec<BenchModel>, BenchError> {
    if args.weights == 0 {
        return Err(BenchError::Args("--weights must be at least 1".into()));
    }
    let Some(source) = &args.source else {
        check_degree(args.degree, args.nodes)?;
        let mut model = BenchModel::synthetic(args);
        reweight_model(&mut model, args.weights);
        return Ok(vec![model]);
    };
    // clap's `requires` makes this infallible from the CLI, but stay
    // defensive against direct `RunArgs` construction (e.g. from tests).
    let topology = args
        .topology
        .as_ref()
        .ok_or_else(|| BenchError::Corpus("--source requires --topology".to_owned()))?;
    let spec =
        corpus::load_topology_spec(topology).map_err(|e| BenchError::Corpus(e.to_string()))?;
    let records = corpus::load_corpus(source, &spec, args.limit)
        .map_err(|e| BenchError::Corpus(e.to_string()))?;
    Ok(records
        .into_iter()
        .map(|(record, graph)| {
            let mut model = BenchModel {
                graph,
                nonce: record.nonce,
                topology_hash: record.topology_hash,
            };
            reweight_model(&mut model, args.weights);
            model
        })
        .collect())
}

/// Decode a 64-hex nonce as a `ChaCha8` seed, or the all-zero seed for any
/// other string.
fn nonce_seed(nonce: &str) -> [u8; 32] {
    let bytes: Option<Vec<u8>> = nonce
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = char::from(pair[0]).to_digit(16)?;
            let low = char::from(pair[1]).to_digit(16)?;
            u8::try_from(high * 16 + low).ok()
        })
        .collect();
    bytes
        .filter(|_| nonce.len() == 64)
        .and_then(|bytes| bytes.try_into().ok())
        .unwrap_or([0; 32])
}

/// Use the corpus nonce as the `ChaCha8` seed, or the default all-zero seed
/// for a synthetic model without a 64-hex nonce. Draw fields before couplings,
/// using the protocol draw's word-modulo-range convention.
fn reweight_model(model: &mut BenchModel, weights: u32) {
    if weights == 1 {
        return;
    }
    let seed = nonce_seed(&model.nonce);
    let mut rng = ChaCha8Rng::from_seed(seed);
    let field_range = 2 * u64::from(weights) + 1;
    for h in &mut model.graph.h {
        let mut word = u64::from(rng.next_u32());
        if field_range > u64::from(u32::MAX) {
            word |= u64::from(rng.next_u32()) << 32;
        }
        // The draw is at most 2*u32::MAX, which is exact in f64.
        #[allow(clippy::cast_precision_loss)]
        let draw = (word % field_range) as f64;
        *h = draw - f64::from(weights);
    }
    for j in &mut model.graph.j {
        let magnitude = f64::from(rng.next_u32() % weights + 1);
        *j = if *j == 0.0 {
            0.0
        } else {
            j.signum() * magnitude
        };
    }
}

fn backend_name(kernel: KernelKind) -> &'static str {
    match kernel {
        KernelKind::Sa => "cuda-sa",
        KernelKind::Msa => "cuda-msa",
        KernelKind::Gibbs => "cuda-gibbs",
    }
}

fn best_energy_metric(reads: &[crate::SamplerResult]) -> Option<Metric> {
    let best = reads.iter().map(|read| read.energy_milli).min()?;
    #[allow(clippy::cast_precision_loss)] // benchmark energies stay well below 2^53 milli units
    Some(Metric::new(
        "best_energy",
        best as f64 / 1000.0,
        "energy",
        Source::CounterDerived,
    ))
}

/// Per-span open timestamp, stashed in the span's extensions.
struct SpanStart(Instant);

/// `tracing_subscriber::Layer` that sums span busy-time by name into a shared
/// map. `run_bench` reads (and drains) this between measured `bench_many`
/// calls to build each cell's [`HostSpans`] from the Task-2 seam spans,
/// without threading extra return values out of `bench_many`/`sample_one`.
struct SpanAggregator {
    totals: Arc<Mutex<HashMap<String, u64>>>,
}

impl SpanAggregator {
    fn new() -> (Self, Arc<Mutex<HashMap<String, u64>>>) {
        let totals = Arc::new(Mutex::new(HashMap::new()));
        (
            Self {
                totals: Arc::clone(&totals),
            },
            totals,
        )
    }
}

impl<S> Layer<S> for SpanAggregator
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanStart(Instant::now()));
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let busy_ns = span.extensions().get::<SpanStart>().map_or(0, |s| {
            u64::try_from(s.0.elapsed().as_nanos()).unwrap_or(u64::MAX)
        });
        let mut totals = self.totals.lock().unwrap_or_else(PoisonError::into_inner);
        *totals.entry(span.name().to_owned()).or_insert(0) += busy_ns;
    }
}

/// Drain the current per-span totals, resetting the map for the next cell.
fn take_cell(totals: &Arc<Mutex<HashMap<String, u64>>>) -> HashMap<String, u64> {
    let mut guard = totals.lock().unwrap_or_else(PoisonError::into_inner);
    std::mem::take(&mut *guard)
}

/// Open device + identity, threaded through the grid loop.
struct BenchContext<'a> {
    device: &'a CudaDevice,
    kernel: KernelKind,
    nonces: usize,
    backend: &'static str,
    device_name: String,
    totals: Arc<Mutex<HashMap<String, u64>>>,
}

/// `bench run` / `bench fold` entry point the binaries dispatch to.
///
/// # Errors
///
/// [`BenchError::Device`] if the device fails to open or a `bench_many` call
/// fails; [`BenchError::Io`] for any output file/directory failure;
/// [`BenchError::Fold`] if a Nsight report fails to parse; [`BenchError::Corpus`]
/// if `--source`/`--topology` fail to load or a nonce fails to redraw.
pub fn run_bench(
    device_index: usize,
    kernel: KernelKind,
    max_nodes: usize,
    action: &BenchAction,
) -> Result<(), BenchError> {
    match action {
        BenchAction::Run(args) => run_run(device_index, kernel, max_nodes, args),
        BenchAction::Fold(args) => run_fold(args),
    }
}

/// Refuse a `--reads` the kernel would silently clamp. `bench_many` caps reads
/// per nonce at `streaming::max_reads(kernel)`, and a clamped run would
/// report the requested count against the clamped timing.
fn check_reads(kernel: KernelKind, reads: u64) -> Result<(), BenchError> {
    let max_reads = u64::from(crate::streaming::max_reads(kernel));
    if (1..=max_reads).contains(&reads) {
        return Ok(());
    }
    Err(BenchError::Args(format!(
        "--reads {reads} is outside 1..={max_reads} for the {} kernel",
        kernel.name()
    )))
}

fn run_run(
    device_index: usize,
    kernel: KernelKind,
    max_nodes: usize,
    args: &RunArgs,
) -> Result<(), BenchError> {
    if args.msa_path.is_some() && kernel != KernelKind::Msa {
        return Err(BenchError::Args(
            "--msa-path requires the MSA kernel".into(),
        ));
    }
    if args.msa_shared_spins.is_some() && kernel != KernelKind::Msa {
        return Err(BenchError::Args(
            "--msa-shared-spins requires the MSA kernel".into(),
        ));
    }
    std::fs::create_dir_all(&args.out).map_err(|e| BenchError::Io(e.to_string()))?;
    let sweeps: Vec<u64> = if args.sweeps.is_empty() {
        vec![1024]
    } else {
        args.sweeps.clone()
    };
    let models = load_models(args)?;

    let (agg_layer, totals) = SpanAggregator::new();
    let flame = args
        .flame
        .as_ref()
        .map(|p| tracing_flame::FlameLayer::with_file(p).map_err(|e| BenchError::Io(e.to_string())))
        .transpose()?;
    let (flame_layer, flame_guard) = match flame {
        Some((layer, guard)) => (Some(layer), Some(guard)),
        None => (None, None),
    };
    let subscriber = tracing_subscriber::registry()
        .with(agg_layer)
        .with(flame_layer);

    tracing::subscriber::with_default(subscriber, || -> Result<(), BenchError> {
        // Open for the kernel actually being benched, at the requested
        // capacity: the defaulting `open` would compile SA at 5000 and reject
        // any larger graph regardless of which binary is running.
        let mut device = CudaDevice::open_with_nodes(device_index, kernel, max_nodes)
            .map_err(|e| BenchError::Device(e.to_string()))?;
        device.msa_shared_spins_cap = args.msa_shared_spins;
        check_reads(kernel, args.reads)?;
        let width = stream_width(&device, kernel);
        let nonces = usize::try_from(args.nonces).unwrap_or(usize::MAX);
        if nonces > width {
            return Err(BenchError::Args(format!(
                "--nonces {nonces} exceeds the device's stream width of {width}"
            )));
        }
        let device_name = device
            .name()
            .unwrap_or_else(|_| format!("cuda-{device_index}"));
        // The JIT compile happened inside `open()`, under this same
        // subscriber, so its span landed in `totals` before any cell runs.
        let mut jit_ns = take_cell(&totals).get("jit").copied();
        let ctx = BenchContext {
            device: &device,
            kernel,
            nonces,
            backend: backend_name(kernel),
            device_name,
            totals,
        };
        for &num_sweeps in &sweeps {
            run_cell(&ctx, args, num_sweeps, &models, &mut jit_ns)?;
        }
        Ok(())
    })?;

    if let Some(guard) = flame_guard {
        guard.flush().map_err(|e| BenchError::Io(e.to_string()))?;
    }
    Ok(())
}

/// Run every model at one `num_sweeps` grid cell, appending one JSON line
/// per (model, measured repeat) to a single
/// `<out>/<backend>_n<nodes>_s<num_sweeps>.jsonl` file. `nodes` is read off
/// the first model: every model in a `bench run` invocation shares one
/// topology (the synthetic circulant graph's `--nodes`, or `--topology`'s node set).
fn run_cell(
    ctx: &BenchContext<'_>,
    args: &RunArgs,
    num_sweeps: u64,
    models: &[BenchModel],
    jit_ns: &mut Option<u64>,
) -> Result<(), BenchError> {
    let nodes = models
        .first()
        .and_then(|m| u64::try_from(m.graph.num_nodes()).ok())
        .unwrap_or(args.nodes);
    let path = args
        .out
        .join(format!("{}_n{nodes}_s{num_sweeps}.jsonl", ctx.backend));
    let mut file = File::create(&path).map_err(|e| BenchError::Io(e.to_string()))?;
    for model in models {
        run_model(ctx, args, num_sweeps, model, &mut file, jit_ns)?;
    }
    Ok(())
}

/// Warm up, then measure `args.repeats` isolated single-shot runs of one
/// model at one `num_sweeps` grid cell, appending each measured repeat's
/// JSON line to `file`.
fn run_model(
    ctx: &BenchContext<'_>,
    args: &RunArgs,
    num_sweeps: u64,
    model: &BenchModel,
    file: &mut File,
    jit_ns: &mut Option<u64>,
) -> Result<(), BenchError> {
    let graph = &model.graph;
    let params = SampleParams {
        num_reads: usize::try_from(args.reads).unwrap_or(usize::MAX),
        num_sweeps: usize::try_from(num_sweeps).unwrap_or(usize::MAX),
        sweeps_per_beta: usize::try_from(args.sweeps_per_beta).unwrap_or(1),
        ..SampleParams::default()
    };
    let msa_lane = args.msa_path.and_then(MsaPath::lane);
    let plan = bench_plan(ctx.device, graph, ctx.kernel, msa_lane)
        .map_err(|e| BenchError::Device(e.to_string()))?;
    let (beta, sweeps_per_beta) = crate::streaming::build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
        params.schedule,
        plan.beta_factor(),
    );
    let cfg = CellConfig {
        reads: args.reads,
        num_sweeps,
        sweeps_per_beta: u64::try_from(sweeps_per_beta).unwrap_or(1),
        num_betas: u64::try_from(beta.len()).unwrap_or(1),
        nodes: u64::try_from(graph.num_nodes()).unwrap_or(args.nodes),
        edges: u64::try_from(graph.edges.len()).unwrap_or(0),
    };

    for i in 0..(args.warmup + args.repeats) {
        take_cell(&ctx.totals); // clear stale totals before the measured call
        let model_start = Instant::now();
        let (reads, dev) = bench_many(ctx.device, graph, &params, ctx.kernel, msa_lane, ctx.nonces)
            .map_err(|e| BenchError::Device(e.to_string()))?;
        let model_total_ns = u64::try_from(model_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let spans = take_cell(&ctx.totals);
        if i < args.warmup {
            continue;
        }
        let host = HostSpans {
            problem_setup_ns: spans.get("problem_setup").copied().unwrap_or(0),
            beta_build_ns: spans.get("beta_build").copied().unwrap_or(0),
            energy_score_ns: spans.get("energy_score").copied().unwrap_or(0),
            jit_ns: jit_ns.take(),
        };
        let mut record = assemble_record(
            ctx.backend,
            &ctx.device_name,
            &model.nonce,
            &model.topology_hash,
            &cfg,
            &dev,
            &host,
            model_total_ns,
        );
        record.metrics.extend(best_energy_metric(&reads));
        #[allow(clippy::cast_precision_loss)] // counts far below 2^53
        record.metrics.push(Metric::new(
            "concurrent_models",
            ctx.nonces as f64,
            "count",
            Source::Tracing,
        ));
        if ctx.kernel == KernelKind::Msa {
            let layout =
                crate::streaming::msa_layout(ctx.device, graph.num_nodes(), params.num_reads);
            #[allow(clippy::cast_precision_loss)] // counts far below 2^53
            record.metrics.push(Metric::new(
                "shared_spins",
                layout.shared_spins as f64,
                "spins",
                Source::Tracing,
            ));
        }
        let line = serde_json::to_string(&record).map_err(|e| BenchError::Io(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| BenchError::Io(e.to_string()))?;
    }
    Ok(())
}

/// Fold an external Nsight report into an already-emitted host JSONL,
/// writing the folded records to `args.out`. Pure file I/O + parsing — needs
/// no GPU, so this path is headless-testable (see `tests/nsight_parser.rs`).
fn run_fold(args: &FoldArgs) -> Result<(), BenchError> {
    let host_json =
        std::fs::read_to_string(&args.host_json).map_err(|e| BenchError::Io(e.to_string()))?;
    let kern_csv =
        std::fs::read_to_string(&args.nsys_kern_sum).map_err(|e| BenchError::Io(e.to_string()))?;
    let kern_rows =
        nsight::parse_nsys_kern_sum(&kern_csv).map_err(|e| BenchError::Fold(e.to_string()))?;

    let per_sweep_ns = fold_per_sweep(args)?;
    let ncu_metrics = args
        .ncu_csv
        .as_ref()
        .map(|p| {
            let csv = std::fs::read_to_string(p).map_err(|e| BenchError::Io(e.to_string()))?;
            nsight::parse_ncu_csv(&csv).map_err(|e| BenchError::Fold(e.to_string()))
        })
        .transpose()?;

    let mut out = String::new();
    for line in host_json.lines().filter(|l| !l.trim().is_empty()) {
        let mut record: BenchRecord =
            serde_json::from_str(line).map_err(|e| BenchError::Fold(e.to_string()))?;
        nsight::fold_into(
            &mut record,
            &kern_rows,
            ncu_metrics.as_deref(),
            per_sweep_ns,
        );
        out.push_str(&serde_json::to_string(&record).map_err(|e| BenchError::Fold(e.to_string()))?);
        out.push('\n');
    }
    std::fs::write(&args.out, out).map_err(|e| BenchError::Io(e.to_string()))?;
    Ok(())
}

/// The two-point per-sweep derivation input, if `args` supplies a complete
/// (hi CSV, sweeps-lo, sweeps-hi) triple; `None` skips the `sweep` overwrite.
fn fold_per_sweep(args: &FoldArgs) -> Result<Option<u64>, BenchError> {
    let (Some(hi_path), Some(sweeps_lo), Some(sweeps_hi)) =
        (&args.kern_sum_hi, args.sweeps_lo, args.sweeps_hi)
    else {
        return Ok(None);
    };
    let lo_csv =
        std::fs::read_to_string(&args.nsys_kern_sum).map_err(|e| BenchError::Io(e.to_string()))?;
    let lo_rows =
        nsight::parse_nsys_kern_sum(&lo_csv).map_err(|e| BenchError::Fold(e.to_string()))?;
    let hi_csv = std::fs::read_to_string(hi_path).map_err(|e| BenchError::Io(e.to_string()))?;
    let hi_rows =
        nsight::parse_nsys_kern_sum(&hi_csv).map_err(|e| BenchError::Fold(e.to_string()))?;
    let lo_avg = lo_rows
        .first()
        .ok_or_else(|| BenchError::Fold("no kernel rows in --nsys-kern-sum".into()))?
        .avg_ns;
    let hi_avg = hi_rows
        .first()
        .ok_or_else(|| BenchError::Fold("no kernel rows in --kern-sum-hi".into()))?
        .avg_ns;
    let s_lo = sweeps_lo * args.sweeps_per_beta.max(1);
    let s_hi = sweeps_hi * args.sweeps_per_beta.max(1);
    Ok(Some(nsight::derive_per_sweep(lo_avg, s_lo, hi_avg, s_hi)))
}

#[cfg(test)]
mod tests {
    use super::{
        assemble_record, check_degree, check_reads, circulant_graph, CellConfig, HostSpans,
        SpanAggregator,
    };
    use crate::schema::Scope;
    use crate::streaming::DeviceTimings;
    use crate::KernelKind;
    use std::sync::Arc;
    use tracing::info_span;
    use tracing_subscriber::prelude::*;

    #[derive(clap::Parser, Debug)]
    struct RunCli {
        #[command(flatten)]
        args: super::RunArgs,
    }

    #[test]
    fn circulant_degree_two_is_the_old_ring() {
        let g = circulant_graph(5, 2, None);
        assert_eq!(&*g.edges, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    }

    #[test]
    fn circulant_degree_forty_has_n_times_twenty_edges_and_degree_forty() {
        let g = circulant_graph(1000, 40, None);
        assert_eq!(g.edges.len(), 20_000);
        let topo = crate::topology::SelfFeedingTopology::build(&g);
        assert_eq!(topo.max_degree(), 40);
    }

    fn sorted_degrees(g: &crate::IsingGraph) -> Vec<usize> {
        let mut deg = vec![0; g.num_nodes()];
        for &(u, v) in g.edges.iter() {
            deg[u] += 1;
            deg[v] += 1;
        }
        deg.sort_unstable();
        deg
    }

    #[test]
    fn shuffle_relabels_without_changing_the_degree_sequence() {
        let plain = circulant_graph(500, 20, None);
        for seed in [[7; 32], [0; 32]] {
            let shuffled = circulant_graph(500, 20, Some(seed));
            assert_eq!(plain.edges.len(), shuffled.edges.len());
            assert_ne!(plain.edges, shuffled.edges, "seed {seed:?} is the identity");
            assert_eq!(sorted_degrees(&plain), sorted_degrees(&shuffled));
        }
    }

    #[test]
    fn unbuildable_degrees_are_refused() {
        assert!(check_degree(3, 100).is_err(), "odd");
        assert!(check_degree(0, 100).is_err(), "zero");
        assert!(check_degree(100, 100).is_err(), "degree equals nodes");
        assert!(check_degree(6, 6).is_err(), "degree above nodes");
        assert!(check_degree(64, 1000).is_err(), "above the msa limit");
        assert!(check_degree(1_000_000, 200_000).is_err(), "huge");
        assert!(check_degree(40, 1000).is_ok());
        assert!(check_degree(62, 1000).is_ok());
        assert!(check_degree(2, 5).is_ok());
    }

    #[test]
    fn degree_and_shuffle_conflict_with_source() {
        use clap::Parser as _;
        for flag in [&["--degree", "4"][..], &["--shuffle"][..]] {
            let mut argv = vec!["bench", "--out", "x", "--source", "s", "--topology", "t"];
            argv.extend_from_slice(flag);
            assert!(RunCli::try_parse_from(argv).is_err(), "{flag:?}");
        }
    }

    #[test]
    fn weights_one_preserves_synthetic_graph() {
        use clap::Parser as _;
        let args = RunCli::try_parse_from(["bench", "--out", "unused", "--weights", "1"])
            .unwrap()
            .args;
        let models = super::load_models(&args).unwrap();
        let original = super::circulant_graph(args.nodes, 2, None);
        assert_eq!(models[0].graph.h, original.h);
        assert_eq!(models[0].graph.j, original.j);
        assert_eq!(models[0].graph.edges, original.edges);
    }

    #[test]
    fn weights_are_bounded_and_deterministic_per_nonce() {
        use clap::Parser as _;
        let args = RunCli::try_parse_from([
            "bench",
            "--out",
            "unused",
            "--weights",
            "5",
            "--nonce",
            "b4179357b751254ed0e68b5e969dcb50e73fd8c56be192b79d286ff2722d6a72",
        ])
        .unwrap()
        .args;
        let first = super::load_models(&args).unwrap().remove(0).graph;
        let second = super::load_models(&args).unwrap().remove(0).graph;
        assert_eq!(first.h, second.h);
        assert_eq!(first.j, second.j);
        assert_eq!(first.edges, second.edges);
        assert!(first
            .h
            .iter()
            .all(|h| (-5.0..=5.0).contains(h) && h.fract() == 0.0));
        assert!(first
            .j
            .iter()
            .all(|j| (1.0..=5.0).contains(j) && j.fract() == 0.0));
        assert!(first.h.iter().any(|h| *h != 0.0));
        assert!(first.j.iter().any(|j| *j > 1.0));
        let mut other = args;
        other.nonce = Some("00".repeat(32));
        let other = super::load_models(&other).unwrap().remove(0).graph;
        assert_ne!(first.h, other.h);
    }

    #[test]
    fn weights_cli_defaults_and_rejects_zero() {
        use clap::Parser as _;
        assert!(RunCli::try_parse_from(["bench", "--out", "unused", "--weights", "0"]).is_err());
        let default = RunCli::try_parse_from(["bench", "--out", "unused"])
            .unwrap()
            .args;
        let explicit = RunCli::try_parse_from(["bench", "--out", "unused", "--weights", "1"])
            .unwrap()
            .args;
        assert_eq!(default.weights, 1);
        let mut invalid = default.clone();
        invalid.weights = 0;
        assert!(matches!(
            super::load_models(&invalid),
            Err(super::BenchError::Args(_))
        ));
        assert_eq!(
            super::load_models(&default).unwrap()[0].graph.h,
            super::load_models(&explicit).unwrap()[0].graph.h
        );
    }

    #[test]
    fn weights_large_range_draws_fields_of_both_signs() {
        use clap::Parser as _;
        let args = RunCli::try_parse_from(["bench", "--out", "unused", "--weights", "4294967295"])
            .unwrap()
            .args;
        let graph = super::load_models(&args).unwrap().remove(0).graph;
        assert!(graph.h.iter().any(|h| *h > 0.0));
        assert!(graph.h.iter().any(|h| *h < 0.0));
        assert!(graph.h.iter().all(|h| h.abs() <= f64::from(u32::MAX)));
    }

    #[test]
    fn weights_preserve_corpus_signs_and_identity() {
        use clap::Parser as _;
        let dir = std::env::temp_dir().join(format!("bench-weights-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let topology = dir.join("topology.json");
        let source = dir.join("corpus.jsonl");
        std::fs::write(&topology, r#"{"nodes":[0,1,2,3],"edges":[[0,1],[1,2],[2,3],[0,3]],"allowed_h_milli":[-1000,0,1000],"allowed_j_milli":[-1000,1000]}"#).unwrap();
        std::fs::write(
            &source,
            "{\"nonce\":\"b4179357b751254ed0e68b5e969dcb50e73fd8c56be192b79d286ff2722d6a72\"}\n",
        )
        .unwrap();
        let mut args = RunCli::try_parse_from([
            "bench",
            "--out",
            "unused",
            "--source",
            source.to_str().unwrap(),
            "--topology",
            topology.to_str().unwrap(),
            "--weights",
            "1",
        ])
        .unwrap()
        .args;
        let original = super::load_models(&args).unwrap().remove(0);
        assert_eq!(original.graph.h, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(original.graph.j, vec![1.0, -1.0, -1.0, 1.0]);
        args.weights = 5;
        let weighted = super::load_models(&args).unwrap().remove(0);
        assert_eq!(weighted.graph.edges, original.graph.edges);
        assert_eq!(weighted.nonce, original.nonce);
        assert_eq!(weighted.topology_hash, original.topology_hash);
        for (&j, &original_j) in weighted.graph.j.iter().zip(&original.graph.j) {
            assert_eq!(j.is_sign_positive(), original_j.is_sign_positive());
            assert!((1.0..=5.0).contains(&j.abs()));
        }
        assert!(weighted.graph.h.iter().all(|h| (-5.0..=5.0).contains(h)));
        let repeated = super::load_models(&args).unwrap().remove(0);
        assert_eq!(weighted.graph.h, repeated.graph.h);
        assert_eq!(weighted.graph.j, repeated.graph.j);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn best_energy_metric_uses_lowest_read_in_energy_units() {
        let reads: Vec<_> = [2000, -3500, -1000]
            .into_iter()
            .map(|energy_milli| crate::SamplerResult {
                spins: vec![1],
                energy_milli,
            })
            .collect();
        let metric = super::best_energy_metric(&reads).unwrap();
        assert_eq!(metric.name, "best_energy");
        assert_eq!(metric.value.to_bits(), (-3.5_f64).to_bits());
        assert_eq!(metric.unit, "energy");
        assert_eq!(metric.source, crate::schema::Source::CounterDerived);
        assert!(super::best_energy_metric(&[]).is_none());
    }

    #[test]
    fn msa_path_cli_parses_all_lanes() {
        use clap::Parser as _;
        for path in ["auto", "weighted-i32", "weighted-f32"] {
            let parsed = RunCli::try_parse_from(["bench", "--out", "unused", "--msa-path", path]);
            assert!(parsed.is_ok(), "{path}: {parsed:?}");
            let selected = parsed.unwrap().args.msa_path.unwrap();
            let expected = match path {
                "weighted-i32" => Some(crate::cuda_device::MsaLane::I32),
                "weighted-f32" => Some(crate::cuda_device::MsaLane::F32),
                _ => None,
            };
            assert_eq!(selected.lane(), expected);
        }
        assert!(
            RunCli::try_parse_from(["bench", "--out", "unused", "--msa-path", "invalid"]).is_err()
        );
        assert_eq!(
            RunCli::try_parse_from(["bench", "--out", "unused"])
                .unwrap()
                .args
                .msa_path,
            None
        );
    }

    #[test]
    fn msa_path_flag_rejected_for_other_kernels() {
        use clap::Parser as _;
        for path in ["auto", "weighted-i32", "weighted-f32"] {
            let args = RunCli::try_parse_from(["bench", "--out", "unused", "--msa-path", path])
                .unwrap()
                .args;
            for kernel in [KernelKind::Sa, KernelKind::Gibbs] {
                let err = super::run_bench(
                    usize::MAX,
                    kernel,
                    0,
                    &super::BenchAction::Run(args.clone()),
                )
                .unwrap_err();
                assert!(
                    err.to_string()
                        .contains("--msa-path requires the MSA kernel"),
                    "{err}"
                );
            }
        }
    }

    #[test]
    fn assemble_record_maps_timings_to_contract_parts() {
        let t = DeviceTimings {
            kernel_ns: 800_000,
            upload_ns: 120_000,
            download_ns: 60_000,
            poll_wait_ns: 5_000,
        };
        let cfg = CellConfig {
            reads: 8,
            num_sweeps: 1024,
            sweeps_per_beta: 4,
            num_betas: 256,
            nodes: 64,
            edges: 64,
        };
        let host = HostSpans {
            problem_setup_ns: 10_000,
            beta_build_ns: 2_000,
            energy_score_ns: 40_000,
            jit_ns: None,
        };
        let rec = assemble_record("cuda-sa", "TestGPU", "0x1", "0xa", &cfg, &t, &host, 900_000);
        let sweep = rec.parts.iter().find(|p| p.part == "sweep").unwrap();
        assert_eq!(sweep.scope, Scope::Device);
        assert_eq!(sweep.count, 1024); // S_total = num_betas * sweeps_per_beta = 256 * 4
        assert!((sweep.per_call_ns - 800_000.0 / 1024.0).abs() < 1e-9);
        let energy = rec.parts.iter().find(|p| p.part == "energy_score").unwrap();
        assert_eq!(energy.count, 8); // one score per read
                                     // residual excludes device parts (launch, sweep).
        assert!(rec.residual_ns <= 900_000);
        assert!(rec.metrics.is_empty(), "no Nsight fold ran yet");
    }

    #[test]
    fn assemble_record_includes_jit_only_when_supplied() {
        let t = DeviceTimings::default();
        let cfg = CellConfig {
            reads: 1,
            num_sweeps: 1,
            sweeps_per_beta: 1,
            num_betas: 1,
            nodes: 1,
            edges: 0,
        };
        let no_jit = HostSpans {
            problem_setup_ns: 0,
            beta_build_ns: 0,
            energy_score_ns: 0,
            jit_ns: None,
        };
        let rec = assemble_record("cuda-sa", "g", "n", "t", &cfg, &t, &no_jit, 0);
        assert!(!rec.parts.iter().any(|p| p.part == "jit"));

        let with_jit = HostSpans {
            jit_ns: Some(500_000),
            ..no_jit
        };
        let rec = assemble_record("cuda-sa", "g", "n", "t", &cfg, &t, &with_jit, 500_000);
        assert!(rec.parts.iter().any(|p| p.part == "jit"));
    }

    #[test]
    fn span_aggregator_sums_busy_time_by_name_across_repeats() {
        let (layer, totals) = SpanAggregator::new();
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            for _ in 0..3 {
                let _s = info_span!("unit_part").entered();
                std::hint::black_box(0u64);
            }
        });
        let snap = Arc::clone(&totals);
        let guard = snap.lock().unwrap();
        assert!(guard.contains_key("unit_part"));
        assert!(*guard.get("unit_part").unwrap() > 0);
    }

    #[test]
    fn take_cell_drains_and_resets_between_calls() {
        let (layer, totals) = SpanAggregator::new();
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let _s = info_span!("seam").entered();
        });
        let first = super::take_cell(&totals);
        assert!(first.contains_key("seam"));
        let second = super::take_cell(&totals);
        assert!(second.is_empty(), "totals must reset after take_cell");
    }

    #[test]
    fn check_reads_refuses_counts_the_kernel_would_clamp() {
        for (kernel, cap) in [
            (KernelKind::Sa, 256),
            (KernelKind::Msa, 128),
            (KernelKind::Gibbs, 256),
        ] {
            assert!(check_reads(kernel, 1).is_ok(), "{kernel:?} 1");
            assert!(check_reads(kernel, cap).is_ok(), "{kernel:?} cap");
            assert!(check_reads(kernel, 0).is_err(), "{kernel:?} 0");
            assert!(check_reads(kernel, cap + 1).is_err(), "{kernel:?} cap+1");
        }
        let err = check_reads(KernelKind::Msa, 256).unwrap_err().to_string();
        assert_eq!(
            err,
            "bench args: --reads 256 is outside 1..=128 for the msa kernel"
        );
    }
}

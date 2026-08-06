//! `bench` subcommand: isolated single-model timing + unified JSON emission.
//!
//! Runs the isolated single-shot launch path ([`crate::streaming::bench_one`])
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
//! sm__warps_active.avg.pct_of_peak_sustained_active \
//!     --kernel-name regex:'cuda_(sa|gibbs)_self_feeding' \
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
use crate::cuda_device::CudaDevice;
use crate::nsight;
use crate::schema::{BenchRecord, Part, Scope, Source};
use crate::streaming::{bench_one, DeviceTimings};
use crate::{Algorithm, IsingGraph, SampleParams};
use clap::{Args, Subcommand};
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

/// Arguments for `bench run`.
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// Reads per model.
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
    /// Node count of the synthetic ring problem (`h=0`, `j=1`, ring edges).
    /// Ignored when `--source` is given (the real corpus topology's node
    /// count is used instead). Useful on its own for micro checks.
    #[arg(long, default_value_t = 64)]
    pub nodes: u64,
    /// Corpus JSONL to bench real models (one `{"nonce":"<64hex>", ...}`
    /// line per model, unknown fields ignored); requires `--topology`.
    /// Redraws each nonce's real `(h, J)` and benches that graph instead of
    /// the synthetic ring.
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
    /// no real nonce).
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

/// A deterministic synthetic ring Ising problem (`h=0`, `j=1`, ring edges),
/// used when `bench run` is not given `--source`/`--topology`. Not
/// timing-representative of the real corpus topology (degree 2 vs. ~18) —
/// useful for micro checks only.
#[allow(clippy::cast_possible_truncation)] // n bounded well under u32::MAX by CLI/kernel limits
fn ring_graph(n: u64) -> IsingGraph {
    let n = n as usize;
    let h = vec![0.0; n];
    let j = vec![1.0; n];
    let edges = (0..n).map(|i| (i, (i + 1) % n.max(1))).collect();
    IsingGraph::new(h, j, edges)
}

/// One graph to bench in the grid: either the synthetic ring problem or a
/// `--source` corpus record redrawn from its nonce. All models in one `bench
/// run` invocation share a topology (the synthetic ring's `--nodes`, or
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
    /// The synthetic ring model, tagged with `--nonce`/`--topology-hash`
    /// metadata passthrough (synthetic problems have no real identity).
    fn synthetic(args: &RunArgs) -> Self {
        Self {
            graph: ring_graph(args.nodes),
            nonce: args.nonce.clone().unwrap_or_else(|| "0x0".to_owned()),
            topology_hash: args
                .topology_hash
                .clone()
                .unwrap_or_else(|| "0x0".to_owned()),
        }
    }
}

/// Build the bench grid's models: a `--source` corpus (redrawn per nonce
/// against `--topology`) when given, else the single synthetic ring problem.
fn load_models(args: &RunArgs) -> Result<Vec<BenchModel>, BenchError> {
    let Some(source) = &args.source else {
        return Ok(vec![BenchModel::synthetic(args)]);
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
        .map(|(record, graph)| BenchModel {
            graph,
            nonce: record.nonce,
            topology_hash: record.topology_hash,
        })
        .collect())
}

fn backend_name(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::Sa => "cuda-sa",
        Algorithm::Gibbs => "cuda-gibbs",
    }
}

/// Per-span open timestamp, stashed in the span's extensions.
struct SpanStart(Instant);

/// `tracing_subscriber::Layer` that sums span busy-time by name into a shared
/// map. `run_bench` reads (and drains) this between measured `bench_one`
/// calls to build each cell's [`HostSpans`] from the Task-2 seam spans,
/// without threading extra return values out of `bench_one`/`sample_one`.
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
    algorithm: Algorithm,
    backend: &'static str,
    device_name: String,
    totals: Arc<Mutex<HashMap<String, u64>>>,
}

/// `bench run` / `bench fold` entry point the binaries dispatch to.
///
/// # Errors
///
/// [`BenchError::Device`] if the device fails to open or a `bench_one` call
/// fails; [`BenchError::Io`] for any output file/directory failure;
/// [`BenchError::Fold`] if a Nsight report fails to parse; [`BenchError::Corpus`]
/// if `--source`/`--topology` fail to load or a nonce fails to redraw.
pub fn run_bench(
    device_index: usize,
    algorithm: Algorithm,
    max_nodes: usize,
    action: &BenchAction,
) -> Result<(), BenchError> {
    match action {
        BenchAction::Run(args) => run_run(device_index, algorithm, max_nodes, args),
        BenchAction::Fold(args) => run_fold(args),
    }
}

fn run_run(
    device_index: usize,
    algorithm: Algorithm,
    max_nodes: usize,
    args: &RunArgs,
) -> Result<(), BenchError> {
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
        // Open for the algorithm actually being benched, at the requested
        // capacity: the defaulting `open` would compile SA at 5000 and reject
        // any larger graph regardless of which binary is running.
        let device = CudaDevice::open_with_nodes(device_index, algorithm, max_nodes)
            .map_err(|e| BenchError::Device(e.to_string()))?;
        let device_name = device
            .name()
            .unwrap_or_else(|_| format!("cuda-{device_index}"));
        // The JIT compile happened inside `open()`, under this same
        // subscriber, so its span landed in `totals` before any cell runs.
        let mut jit_ns = take_cell(&totals).get("jit").copied();
        let ctx = BenchContext {
            device: &device,
            algorithm,
            backend: backend_name(algorithm),
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
/// topology (the synthetic ring's `--nodes`, or `--topology`'s node set).
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
    let (beta, sweeps_per_beta) = crate::streaming::build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
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
        let (_reads, dev) = bench_one(ctx.device, graph, &params, ctx.algorithm)
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
        let record = assemble_record(
            ctx.backend,
            &ctx.device_name,
            &model.nonce,
            &model.topology_hash,
            &cfg,
            &dev,
            &host,
            model_total_ns,
        );
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
    use super::{assemble_record, CellConfig, HostSpans, SpanAggregator};
    use crate::schema::Scope;
    use crate::streaming::DeviceTimings;
    use std::sync::Arc;
    use tracing::info_span;
    use tracing_subscriber::prelude::*;

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
}

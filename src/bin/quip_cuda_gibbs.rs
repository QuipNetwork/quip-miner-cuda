//! CUDA heat-bath Gibbs miner (`quip-cuda-gibbs`).
//!
//! One process per GPU: `--device N` binds CUDA device N and defaults
//! `--miner-id` to `cuda-N` (matching `[cuda.N]` config sections).

use clap::{Parser, Subcommand};
use quip_miner_cuda::bench::{run_bench, BenchAction};
use quip_miner_cuda::capacity::advertised_nodes;
use quip_miner_cuda::capacity::GIBBS_DEFAULT_NODES;
use quip_miner_cuda::cuda_device::{device_label_mismatch, CudaDevice};
use quip_miner_cuda::nvml_gov::UtilGovernor;
use quip_miner_cuda::{cuda_gibbs_identity, CudaSampler, KernelKind};
use quip_solver_core::{run, CommonArgs, OpenError};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    common: CommonArgs,
    /// CUDA device index (one process per GPU). Default 0 → miner id `cuda-0`.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Ceiling percent (1–100) the NVML governor compares foreign GPU load
    /// against. Only consulted with --yielding. The default of 100 never
    /// throttles.
    #[arg(long, default_value_t = 100)]
    utilization: u32,
    /// Share the GPU: end the current session when load from *other* processes
    /// exceeds --utilization. Load from this miner is excluded.
    #[arg(long, default_value_t = false)]
    yielding: bool,
    /// Largest node count to compile the kernel for. The miner opens its
    /// device before the coordinator sends a topology, so this cannot come
    /// from the wire. Values below the default are raised to it; a value
    /// above the algorithm limit or the device's shared-memory budget is an
    /// error at open, never a silent clamp.
    #[arg(long, default_value_t = GIBBS_DEFAULT_NODES)]
    max_nodes: usize,
}

/// Top-level subcommands. Absent → the ordinary coordinator-driven miner
/// session (unchanged behavior).
#[derive(Subcommand)]
enum Command {
    /// Fine-grained per-part timing for one model (isolated single-shot
    /// launch); see `quip-cuda-gibbs bench run --help` / `bench fold --help`.
    #[command(subcommand)]
    Bench(BenchAction),
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();
    if let Some(Command::Bench(action)) = &cli.command {
        return match run_bench(cli.device, KernelKind::Gibbs, cli.max_nodes, action) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("bench failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    // Install a log sink before anything can fail. `quip-solver-core` reports
    // every session error, including `--check` failures and a refused node
    // capacity, through `tracing::error!`, and with no subscriber those
    // records are dropped: the process exits non-zero having printed nothing.
    // The bench path above sets its own scoped subscriber, so this stays off
    // that branch.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.common.log_level));
    // `try_init` fails only when a subscriber is already installed, which is
    // not an error here.
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init(),
    );

    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("cuda-{}", cli.device));
    }

    // Advisory only. The miner id is a wire label; the coordinator owns device
    // selection and passes it as `--device N`. A disagreement means the caller
    // sent no `--device` or sent a different one, and the process is about to
    // mine on a GPU its label does not name — invisible from the coordinator
    // side, which sees only the label.
    if let Some(labelled) = device_label_mismatch(cli.common.miner_id.as_deref(), cli.device) {
        tracing::warn!(
            device = cli.device,
            miner_id = %cli.common.miner_id.as_deref().unwrap_or_default(),
            "miner id names CUDA device {labelled} but --device selects {}; the label is \
             advisory and does not select the device. If the caller is quip-coordinator, it \
             predates the --device flag and needs upgrading",
            cli.device
        );
    }
    run(
        cuda_gibbs_identity(advertised_nodes(KernelKind::Gibbs, cli.max_nodes)),
        &cli.common,
        || {
            let device = CudaDevice::open_with_nodes(cli.device, KernelKind::Gibbs, cli.max_nodes)
                .map_err(|e| OpenError(format!("device {}: {e}", cli.device)))?;
            // The governor binds by PCI bus id, not by ordinal: NVML's index
            // space is PCI-ordered and does not track the CUDA ordinal this
            // process opened. Taking it from the opened device means both
            // APIs name the same physical GPU by construction.
            let gov = UtilGovernor::start(&device.pci_bus_id, cli.utilization, cli.yielding);
            Ok(CudaSampler::new(device, gov, KernelKind::Gibbs))
        },
    )
}

#[cfg(test)]
mod cli_tests {
    use super::{BenchAction, Cli, Command};
    use clap::Parser;

    #[test]
    fn parses_max_nodes() {
        let cli = Cli::parse_from(["quip-cuda-gibbs", "--max-nodes", "5640"]);
        assert_eq!(cli.max_nodes, 5640);
    }

    /// Absent flag must land on the shipped kernel array size, so a plain
    /// invocation compiles exactly what it always did.
    #[test]
    fn max_nodes_defaults_to_the_algorithm_floor() {
        let cli = Cli::parse_from(["quip-cuda-gibbs"]);
        assert_eq!(
            cli.max_nodes,
            quip_miner_cuda::capacity::GIBBS_DEFAULT_NODES
        );
    }

    #[test]
    fn parses_plain_session_invocation() {
        let cli = Cli::parse_from(["quip-cuda-gibbs", "--device", "1"]);
        assert!(cli.command.is_none());
        assert_eq!(cli.device, 1);
    }

    #[test]
    fn parses_bench_run_subcommand() {
        let cli = Cli::parse_from([
            "quip-cuda-gibbs",
            "bench",
            "run",
            "--nodes",
            "64",
            "--sweeps",
            "1024",
            "--out",
            "/tmp/x",
        ]);
        assert!(matches!(cli.command, Some(Command::Bench(_))));
    }

    #[test]
    fn parses_bench_fold_subcommand() {
        let cli = Cli::parse_from([
            "quip-cuda-gibbs",
            "bench",
            "fold",
            "--host-json",
            "/tmp/host.jsonl",
            "--nsys-kern-sum",
            "/tmp/kern.csv",
            "--out",
            "/tmp/folded.jsonl",
        ]);
        assert!(matches!(cli.command, Some(Command::Bench(_))));
    }

    #[test]
    fn parses_bench_run_with_source_topology_and_limit() {
        let cli = Cli::parse_from([
            "quip-cuda-gibbs",
            "bench",
            "run",
            "--source",
            "/tmp/corpus.jsonl",
            "--topology",
            "/tmp/spec.json",
            "--limit",
            "2",
            "--out",
            "/tmp/x",
        ]);
        let Some(Command::Bench(BenchAction::Run(args))) = cli.command else {
            panic!("expected a parsed `bench run`");
        };
        assert_eq!(
            args.source.as_deref(),
            Some(std::path::Path::new("/tmp/corpus.jsonl"))
        );
        assert_eq!(
            args.topology.as_deref(),
            Some(std::path::Path::new("/tmp/spec.json"))
        );
        assert_eq!(args.limit, Some(2));
    }

    #[test]
    fn bench_run_source_without_topology_fails_to_parse() {
        let result = Cli::try_parse_from([
            "quip-cuda-gibbs",
            "bench",
            "run",
            "--source",
            "/tmp/corpus.jsonl",
            "--out",
            "/tmp/x",
        ]);
        assert!(
            result.is_err(),
            "clap `requires` must reject --source without --topology"
        );
    }
}

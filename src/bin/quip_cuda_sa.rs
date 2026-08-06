//! CUDA simulated-annealing miner (`quip-cuda-sa`).
//!
//! One process per GPU: `--device N` binds CUDA device N and defaults
//! `--miner-id` to `cuda-N` (matching `[cuda.N]` config sections).

use clap::{Parser, Subcommand};
use quip_miner_core::{run, CommonArgs, OpenError};
use quip_miner_cuda::bench::{run_bench, BenchAction};
use quip_miner_cuda::capacity::advertised_nodes;
use quip_miner_cuda::capacity::SA_DEFAULT_NODES;
use quip_miner_cuda::cuda_device::CudaDevice;
use quip_miner_cuda::nvml_gov::UtilGovernor;
use quip_miner_cuda::{cuda_sa_identity, Algorithm, CudaSampler};
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
    /// Target GPU utilization ceiling percent (1–100). Used by NVML governor.
    #[arg(long, default_value_t = 100)]
    utilization: u32,
    /// Yield to other GPU users when NVML util exceeds 90%.
    #[arg(long, default_value_t = false)]
    yielding: bool,
    /// Largest node count to compile the kernel for. The miner opens its
    /// device before the coordinator sends a topology, so this cannot come
    /// from the wire. Values below the default are raised to it; a value
    /// above the algorithm limit or the device's shared-memory budget is an
    /// error at open, never a silent clamp.
    #[arg(long, default_value_t = SA_DEFAULT_NODES)]
    max_nodes: usize,
}

/// Top-level subcommands. Absent → the ordinary coordinator-driven miner
/// session (unchanged behavior).
#[derive(Subcommand)]
enum Command {
    /// Fine-grained per-part timing for one model (isolated single-shot
    /// launch); see `quip-cuda-sa bench run --help` / `bench fold --help`.
    #[command(subcommand)]
    Bench(BenchAction),
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();
    if let Some(Command::Bench(action)) = &cli.command {
        return match run_bench(cli.device, Algorithm::Sa, cli.max_nodes, action) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("bench failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("cuda-{}", cli.device));
    }
    run(
        cuda_sa_identity(advertised_nodes(Algorithm::Sa, cli.max_nodes)),
        &cli.common,
        || {
            let device = CudaDevice::open_with_nodes(cli.device, Algorithm::Sa, cli.max_nodes)
                .map_err(|e| OpenError(format!("device {}: {e}", cli.device)))?;
            // NVML device index is u32; CLI takes usize to match CudaDevice::open.
            let nvml_index = u32::try_from(cli.device).map_err(|_| {
                OpenError(format!(
                    "device index {} exceeds u32 range for NVML",
                    cli.device
                ))
            })?;
            let gov = UtilGovernor::start(nvml_index, cli.utilization, cli.yielding);
            Ok(CudaSampler::new(device, gov, Algorithm::Sa))
        },
    )
}

#[cfg(test)]
mod cli_tests {
    use super::{BenchAction, Cli, Command};
    use clap::Parser;

    #[test]
    fn parses_max_nodes() {
        let cli = Cli::parse_from(["quip-cuda-sa", "--max-nodes", "5640"]);
        assert_eq!(cli.max_nodes, 5640);
    }

    /// Absent flag must land on the shipped kernel array size, so a plain
    /// invocation compiles exactly what it always did.
    #[test]
    fn max_nodes_defaults_to_the_algorithm_floor() {
        let cli = Cli::parse_from(["quip-cuda-sa"]);
        assert_eq!(cli.max_nodes, quip_miner_cuda::capacity::SA_DEFAULT_NODES);
    }

    #[test]
    fn parses_plain_session_invocation() {
        let cli = Cli::parse_from(["quip-cuda-sa", "--device", "1"]);
        assert!(cli.command.is_none());
        assert_eq!(cli.device, 1);
    }

    #[test]
    fn parses_bench_run_subcommand() {
        let cli = Cli::parse_from([
            "quip-cuda-sa",
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
            "quip-cuda-sa",
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
            "quip-cuda-sa",
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
            "quip-cuda-sa",
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

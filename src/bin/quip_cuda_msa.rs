//! CUDA multi-spin coded simulated-annealing miner (`quip-cuda-msa`).
//!
//! One process per GPU: `--device N` binds CUDA device N and defaults
//! `--miner-id` to `cuda-N` (matching `[cuda.N]` config sections). Same wire
//! protocol and CLI as `quip-cuda-sa`; select it in the coordinator's
//! `config.toml` with `[cuda.N] binary = "<path>/quip-cuda-msa"`.

use clap::{Parser, Subcommand};
use quip_solver_core::{run, CommonArgs};
use quip_solver_cuda::bench::{run_bench, BenchAction};
use quip_solver_cuda::capacity::{
    instances, msa_device_reads, msa_max_couplers, msa_max_nodes, msa_per_instance_budget, resolve,
    Connectivity, MsaStorage, MSA_DEFAULT_BETAS, MSA_DEFAULT_NODES,
};
use quip_solver_cuda::cuda_device::{device_label_mismatch, CudaDevice};
use quip_solver_cuda::nvml_gov::UtilGovernor;
use quip_solver_cuda::{cuda_msa_identity, KernelKind, MsaSampler};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 2"))]
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
    /// Optional lower cap on the node limit derived from device memory.
    /// Values below 5000 are raised to 5000. A value above the device limit
    /// is an error at open. Without this flag the full device limit applies.
    #[arg(long)]
    max_nodes: Option<usize>,
    /// Print the device capacity grid and exit without starting a session.
    #[arg(long)]
    probe_capacity: bool,
    /// Emit the capacity grid as a single JSON object.
    #[arg(long, requires = "probe_capacity")]
    json: bool,
}

/// Top-level subcommands. Absent → the ordinary coordinator-driven miner
/// session (unchanged behavior).
#[derive(Subcommand)]
enum Command {
    /// Fine-grained per-part timing for one model (isolated single-shot
    /// launch); see `quip-cuda-msa bench run --help` / `bench fold --help`.
    #[command(subcommand)]
    Bench(BenchAction),
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();
    if let Some(Command::Bench(action)) = &cli.command {
        return match run_bench(cli.device, KernelKind::Msa, cli.max_nodes, action) {
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
    // MSA has no compiled node array. Open at the floor to read device
    // limits, then apply the optional cap to the same memory snapshot.
    let mut device =
        match CudaDevice::open_with_nodes(cli.device, KernelKind::Msa, MSA_DEFAULT_NODES) {
            Ok(device) => device,
            Err(e) => {
                tracing::error!("device {}: {e}", cli.device);
                return ExitCode::FAILURE;
            }
        };
    let reads = msa_device_reads(&device.limits);
    if cli.probe_capacity {
        print_probe(&device, reads, cli.json);
        return ExitCode::SUCCESS;
    }
    if let Some(cap) = cli.max_nodes {
        if let Err(e) = resolve(KernelKind::Msa, cap, &device.limits) {
            tracing::error!("device {}: {e}", cli.device);
            return ExitCode::FAILURE;
        }
    }
    let nodes = msa_max_nodes(&device.limits, reads)
        .min(cli.max_nodes.unwrap_or(usize::MAX).max(MSA_DEFAULT_NODES));
    let edges = msa_max_couplers(&device.limits, reads);
    device.max_nodes = nodes;
    let mut identity = cuda_msa_identity(nodes, edges);
    identity.adapt = quip_solver_cuda::cuda_msa_adapt(u32::try_from(reads).unwrap_or(128));
    run(identity, &cli.common, || {
        // The governor binds by PCI bus id, not by ordinal: NVML's index
        // space is PCI-ordered and does not track the CUDA ordinal this
        // process opened. Taking it from the opened device means both
        // APIs name the same physical GPU by construction.
        let gov = UtilGovernor::start(&device.pci_bus_id, cli.utilization, cli.yielding);
        Ok(MsaSampler::new(device, gov))
    })
}

fn print_probe(device: &CudaDevice, reads: usize, json: bool) {
    let limits = &device.limits;
    let cells = quip_solver_cuda::probe::probe(limits, reads);
    let name = device
        .name()
        .unwrap_or_else(|_| format!("cuda-{}", device.device_index));
    let connectivity = |conn| match conn {
        Connectivity::Degree(d) => format!("degree {d}"),
        Connectivity::Complete => "complete".to_owned(),
    };
    if json {
        let cells: Vec<_> = cells.iter().map(|cell| serde_json::json!({
            "connectivity": connectivity(cell.connectivity),
            "precision": format!("{:?}", cell.precision),
            "storage": match cell.storage { MsaStorage::Sparse => "Sparse", MsaStorage::Dense => "Dense" },
            "tier": cell.tier(), "spill": cell.spill,
            "nodes": cell.nodes, "couplers": cell.couplers,
            "per_instance_bytes": cell.per_instance_bytes,
            "expected_slowdown": cell.expected_slowdown,
        })).collect();
        println!(
            "{}",
            serde_json::json!({
                "device_name": name, "sm_count": limits.sm_count,
                "free_bytes": limits.free_bytes, "usable_bytes": limits.usable_bytes(),
                "instances": instances(KernelKind::Msa, limits.sm_count),
                "per_instance_budget": msa_per_instance_budget(limits),
                "reads": reads, "betas": MSA_DEFAULT_BETAS, "cells": cells,
            })
        );
        return;
    }
    println!("{name}: SMs={} free_bytes={} usable_bytes={} instances={} per_instance_budget={} reads={reads} betas={MSA_DEFAULT_BETAS}",
        limits.sm_count, limits.free_bytes, limits.usable_bytes(),
        instances(KernelKind::Msa, limits.sm_count), msa_per_instance_budget(limits));
    println!(
        "{:<12} | {:<48} | {:<48} | {:<48}",
        "Connectivity", "Binary", "Int8", "Float32"
    );
    for row in cells.chunks(3) {
        print!("{:<12}", connectivity(row[0].connectivity));
        for cell in row {
            print!(
                " | {:<48}",
                format!(
                    "{} vars, {} couplers, {}, slowdown unknown",
                    cell.nodes,
                    cell.couplers,
                    cell.tier()
                )
            );
        }
        println!();
    }
    println!("Bytes per instance:");
    println!(
        "{:<12} | {:<16} | {:<16} | {:<16}",
        "Connectivity", "Binary", "Int8", "Float32"
    );
    for row in cells.chunks(3) {
        println!(
            "{:<12} | {:<16} | {:<16} | {:<16}",
            connectivity(row[0].connectivity),
            row[0].per_instance_bytes,
            row[1].per_instance_bytes,
            row[2].per_instance_bytes
        );
    }
    println!("Int32 has the Float32 capacity. Slowdown is unavailable until calibration.");
}

#[cfg(test)]
mod cli_tests {
    use super::{BenchAction, Cli, Command};
    use clap::Parser;

    #[test]
    fn parses_probe_capacity_flags() {
        assert!(Cli::try_parse_from([
            "quip-cuda-msa",
            "--probe-capacity",
            "--device",
            "0",
            "--json"
        ])
        .is_ok());
        assert!(Cli::try_parse_from(["quip-cuda-msa", "--json"]).is_err());
    }

    #[test]
    fn parses_max_nodes() {
        let cli = Cli::parse_from(["quip-cuda-msa", "--max-nodes", "5640"]);
        assert_eq!(cli.max_nodes, Some(5640));
    }

    /// Absent flag lets the device determine the capacity.
    #[test]
    fn max_nodes_defaults_to_the_device_limit() {
        let cli = Cli::parse_from(["quip-cuda-msa"]);
        assert_eq!(cli.max_nodes, None);
    }

    #[test]
    fn parses_plain_session_invocation() {
        let cli = Cli::parse_from(["quip-cuda-msa", "--device", "1"]);
        assert!(cli.command.is_none());
        assert_eq!(cli.device, 1);
    }

    #[test]
    fn parses_dense_bench_flags() {
        let cli = Cli::parse_from([
            "quip-cuda-msa",
            "bench",
            "run",
            "--nodes",
            "200000",
            "--degree",
            "40",
            "--shuffle",
            "--nonces",
            "48",
            "--msa-shared-spins",
            "0",
            "--sweeps",
            "64",
            "--out",
            "/tmp/x",
        ]);
        let Some(Command::Bench(BenchAction::Run(args))) = cli.command else {
            panic!("expected a parsed `bench run`");
        };
        assert_eq!(args.degree, 40);
        assert!(args.shuffle);
        assert_eq!(args.nonces, 48);
        assert_eq!(args.msa_shared_spins, Some(0));
    }

    #[test]
    fn parses_bench_run_subcommand() {
        let cli = Cli::parse_from([
            "quip-cuda-msa",
            "bench",
            "run",
            "--nodes",
            "64",
            "--sweeps",
            "1024",
            "--out",
            "/tmp/x",
        ]);
        let Some(Command::Bench(BenchAction::Run(_))) = cli.command else {
            panic!("expected a parsed `bench run`");
        };
    }

    #[test]
    fn bench_run_source_without_topology_fails_to_parse() {
        let result = Cli::try_parse_from([
            "quip-cuda-msa",
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

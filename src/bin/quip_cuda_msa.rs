//! `quip-cuda-msa`: multi-spin coded SA on CUDA .
//! Same wire protocol and CLI as `quip-cuda-sa`; selected in config.toml with
//! `[cuda.0] binary = "/data/quip-cuda-msa-..."`.
use clap::Parser;
use quip_miner_cuda::capacity::advertised_nodes;
use quip_miner_cuda::capacity::SA_DEFAULT_NODES;
use quip_miner_cuda::cuda_device::{device_label_mismatch, CudaDevice};
use quip_miner_cuda::nvml_gov::UtilGovernor;
use quip_miner_cuda::{cuda_msa_identity, CudaSampler, KernelKind};
use quip_solver_core::{run, CommonArgs, OpenError};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1 msa"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value_t = 100)]
    utilization: u32,
    #[arg(long, default_value_t = false)]
    yielding: bool,
    #[arg(long, default_value_t = SA_DEFAULT_NODES)]
    max_nodes: usize,
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.common.log_level));
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init(),
    );
    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("cuda-{}", cli.device));
    }
    if let Some(labelled) = device_label_mismatch(cli.common.miner_id.as_deref(), cli.device) {
        tracing::warn!(
            device = cli.device,
            "miner id names CUDA device {labelled} but --device selects {}",
            cli.device
        );
    }
    run(
        cuda_msa_identity(advertised_nodes(KernelKind::Msa, cli.max_nodes)),
        &cli.common,
        || {
            let device = CudaDevice::open_with_nodes(cli.device, KernelKind::Msa, cli.max_nodes)
                .map_err(|e| OpenError(format!("device {}: {e}", cli.device)))?;
            let gov = UtilGovernor::start(&device.pci_bus_id, cli.utilization, cli.yielding);
            Ok(CudaSampler::new(device, gov, KernelKind::Msa))
        },
    )
}

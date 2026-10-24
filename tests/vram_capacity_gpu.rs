//! GPU allocation measurements for the MSA capacity estimate.

use cudarc::driver::{sys::CUdevice_attribute, CudaContext};
use quip_solver_cuda::capacity::{msa_footprint, DeviceLimits, ModelShape, MsaStorage, Precision};
use quip_solver_cuda::coefficients;
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::sampler::{sample_ising, SampleError};
use quip_solver_cuda::streaming::{bench_many, bench_many_real, stream_width};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tracing_subscriber::prelude::*;

fn circulant(nodes: usize, degree: usize, precision: Precision) -> IsingGraph {
    let mut edges = Vec::with_capacity(nodes * degree / 2);
    for node in 0..nodes {
        for offset in 1..=degree / 2 {
            edges.push((node, (node + offset) % nodes));
        }
    }
    let j = (0..edges.len())
        .map(|edge| match precision {
            Precision::Int32 if edge % 2 == 0 => 1.001,
            Precision::Int8 if edge % 2 == 0 => 0.016,
            Precision::Int8 => -0.992,
            Precision::Float32 => -std::f64::consts::SQRT_2,
            Precision::Binary | Precision::Int32 => -1.0,
        })
        .collect();
    IsingGraph::new(vec![0.0; nodes], j, edges)
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn oversized_model_refused_before_allocation() {
    let nodes = 50_000;
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, nodes).expect("MSA device");
    let limits = DeviceLimits {
        free_bytes: 64 << 20,
        ..device.limits
    };
    let device = device.with_limits_for_test(limits);
    let ctx = CudaContext::new(device.device_index).expect("session device context");
    let graph = circulant(nodes, 60, Precision::Binary);
    let params = SampleParams {
        num_reads: 128,
        num_sweeps: 1,
        ..SampleParams::default()
    };
    ctx.synchronize().unwrap();
    let (free_before, _) = ctx.mem_get_info().expect("free before refusal");
    let result = sample_ising(&device, &graph, &params, KernelKind::Msa);
    ctx.synchronize().unwrap();
    let (free_after, _) = ctx.mem_get_info().expect("free after refusal");
    match result {
        Err(SampleError::ModelExceedsDeviceMemory {
            nodes: rejected_nodes,
            couplers,
            required_bytes,
            available_bytes,
            instances,
        }) => {
            assert_eq!(rejected_nodes, nodes);
            assert_eq!(couplers, 1_500_000);
            assert!(required_bytes > available_bytes);
            assert_eq!(available_bytes, limits.usable_bytes());
            assert_eq!(instances, device.max_sms);
        }
        Err(other) => panic!("expected ModelExceedsDeviceMemory, got {other:?}"),
        Ok(samples) => panic!(
            "expected ModelExceedsDeviceMemory, got {} successful reads",
            samples.len()
        ),
    }
    eprintln!("refused model: free_before={free_before} free_after={free_after}");
    assert!(
        free_before.saturating_sub(free_after) <= 2 << 20,
        "refused session allocated device memory: before={free_before}, after={free_after}"
    );
}

// Observe live allocations at the existing launch span, before the kernel runs.
// This keeps real models out of the mining stream, which must refuse off-grid data.
struct AllocationAtLaunch {
    ctx: Arc<CudaContext>,
    free: Arc<AtomicUsize>,
}

impl<S> tracing_subscriber::Layer<S> for AllocationAtLaunch
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_enter(&self, id: &tracing::span::Id, context: tracing_subscriber::layer::Context<'_, S>) {
        if context.span(id).unwrap().metadata().name() == "launch" {
            let (free, _) = self.ctx.mem_get_info().expect("free with session alive");
            self.free.store(free, Ordering::SeqCst);
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn footprint_matches_measured_allocation() {
    let nodes = 50_000;
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, nodes).expect("MSA device");
    let ctx = CudaContext::new(device.device_index).expect("session device context");
    let attribute = |key| usize::try_from(ctx.attribute(key).expect("device attribute")).unwrap();
    for degree in [20, 60] {
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let graph = circulant(nodes, degree, precision);
            if precision == Precision::Float32 {
                assert!(coefficients::plan(KernelKind::Msa, &graph, device.max_nodes).is_err());
            } else {
                let plan = coefficients::plan(KernelKind::Msa, &graph, device.max_nodes).unwrap();
                assert_eq!(
                    Precision::from_plan(plan, precision == Precision::Binary),
                    precision
                );
            }
            ctx.synchronize().unwrap();
            let (free_before, _) = ctx.mem_get_info().expect("free before session");
            let limits = DeviceLimits {
                shared_bytes_per_block: attribute(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                ),
                shared_bytes_per_block_optin: attribute(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                ),
                free_bytes: free_before,
                sm_count: device.max_sms,
                threads_per_sm: attribute(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                ),
            };
            let params = SampleParams {
                num_reads: 128,
                num_sweeps: 1,
                ..SampleParams::default()
            };
            let betas = (params.num_sweeps / params.sweeps_per_beta.max(1)).max(1);
            let footprint = msa_footprint(
                ModelShape {
                    nodes,
                    couplers: graph.edges.len(),
                    max_degree: degree,
                },
                precision,
                MsaStorage::Sparse,
                params.num_reads,
                betas,
                &limits,
            );
            let width = stream_width(&device, KernelKind::Msa);
            assert_eq!(footprint.instances, width);
            let free = Arc::new(AtomicUsize::new(0));
            let observer = AllocationAtLaunch {
                ctx: Arc::clone(&ctx),
                free: Arc::clone(&free),
            };
            tracing::subscriber::with_default(
                tracing_subscriber::registry().with(observer),
                || {
                    if precision == Precision::Float32 {
                        bench_many_real(&device, &graph, &params, width).expect("real session");
                    } else {
                        bench_many(&device, &graph, &params, KernelKind::Msa, None, width)
                            .expect("exact session");
                    }
                },
            );
            let free_after = free.load(Ordering::SeqCst);
            assert_ne!(free_after, 0, "launch observer must run");
            let measured = free_before
                .checked_sub(free_after)
                .expect("session consumes memory");
            ctx.synchronize().unwrap();
            eprintln!("degree={degree} precision={precision:?} betas={betas} instances={} measured={measured} footprint={}", footprint.instances, footprint.total());
            assert!(measured > 0);
            assert!(measured <= footprint.total(), "underestimate at degree {degree}, {precision:?}: measured={measured}, footprint={}", footprint.total());
            assert!(
                footprint.total() <= measured + measured / 10 + (64 << 20),
                "overestimate at degree {degree}, {precision:?}: measured={measured}, footprint={}",
                footprint.total()
            );
        }
    }
}

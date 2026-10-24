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
    dense_allocation_cases(device, &ctx);
}

fn dense_allocation_cases(device: CudaDevice, ctx: &Arc<CudaContext>) {
    let mut device = device;
    for nodes in [2_000usize, 8_000] {
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let graph = dense_graph(nodes, precision);
            if nodes == 2_000 || precision == Precision::Binary {
                measure_dense_allocation(&device, ctx, &graph, precision);
            }
            // N=8,000 wide matrices exceed the 48-instance admission budget.
            // This test-only snapshot admits exactly the one instance timed.
            let sms = device.limits.sm_count;
            device.limits.sm_count = 1;
            measure_dense_sweeps(&device, &graph, precision);
            device.limits.sm_count = sms;
        }
    }
}

fn dense_graph(nodes: usize, precision: Precision) -> IsingGraph {
    let mut edges = Vec::with_capacity(nodes * (nodes - 1) / 2);
    let mut j = Vec::with_capacity(edges.capacity());
    let mut rng = 19u64;
    for u in 0..nodes {
        for v in u + 1..nodes {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            edges.push((u, v));
            let sign = if rng & 1 == 0 { -1.0 } else { 1.0 };
            let magnitude = match precision {
                Precision::Binary => 1.0,
                Precision::Int8 => f64::from(u32::try_from(rng % 127 + 1).unwrap()),
                Precision::Int32 => f64::from(u32::try_from(rng % 873 + 128).unwrap()) / 1000.0,
                Precision::Float32 => std::f64::consts::SQRT_2,
            };
            j.push(sign * magnitude);
        }
    }
    let graph = IsingGraph::new(vec![0.0; nodes], j, edges);
    if precision != Precision::Float32 {
        let plan = coefficients::plan(KernelKind::Msa, &graph, nodes).unwrap();
        assert_eq!(Precision::from_plan(plan, true), precision);
    }
    graph
}

fn measure_dense_allocation(
    device: &CudaDevice,
    ctx: &Arc<CudaContext>,
    graph: &IsingGraph,
    precision: Precision,
) {
    let nodes = graph.num_nodes();
    ctx.synchronize().unwrap();
    let (free_before, _) = ctx.mem_get_info().unwrap();
    let footprint = msa_footprint(
        ModelShape {
            nodes,
            couplers: graph.edges.len(),
            max_degree: nodes - 1,
        },
        precision,
        MsaStorage::Dense,
        128,
        1,
        &device.limits,
    );
    let free = Arc::new(AtomicUsize::new(0));
    let observer = AllocationAtLaunch {
        ctx: Arc::clone(ctx),
        free: Arc::clone(&free),
    };
    let p = SampleParams {
        num_reads: 128,
        num_sweeps: 1,
        sweeps_per_beta: 1,
        seed: 19,
        ..SampleParams::default()
    };
    tracing::subscriber::with_default(tracing_subscriber::registry().with(observer), || {
        dense_timing(device, graph, &p, precision, footprint.instances);
    });
    let free_after = free.load(Ordering::SeqCst);
    assert_ne!(free_after, 0);
    let measured = free_before.checked_sub(free_after).unwrap();
    let exact = footprint.total();
    eprintln!(
        "dense {precision:?} N={nodes} instances={} measured={measured} footprint={exact}",
        footprint.instances
    );
    // cuMemGetInfo includes allocator page rounding and pool growth.
    // Allow the same 64 MiB driver overhead as the sparse measurement.
    assert!(measured.abs_diff(exact) <= 64 << 20);
    ctx.synchronize().unwrap();
}

fn dense_timing(
    device: &CudaDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    precision: Precision,
    instances: usize,
) -> quip_solver_cuda::streaming::DeviceTimings {
    if precision == Precision::Float32 {
        bench_many_real(device, graph, params, instances).unwrap().1
    } else {
        bench_many(device, graph, params, KernelKind::Msa, None, instances)
            .unwrap()
            .1
    }
}

// Subtract one-sweep from five-sweep CUDA event times to remove initialization
// and packing. Use the median of three pairs at one instance and 128 reads.
fn measure_dense_sweeps(device: &CudaDevice, graph: &IsingGraph, precision: Precision) {
    let mut differences = Vec::new();
    for _ in 0..3 {
        let mut p = SampleParams {
            num_reads: 128,
            num_sweeps: 1,
            sweeps_per_beta: 1,
            beta_range: Some((0.1, 0.1)),
            seed: 19,
            ..SampleParams::default()
        };
        let one = dense_timing(device, graph, &p, precision, 1);
        p.num_sweeps = 5;
        p.sweeps_per_beta = 5;
        let five = dense_timing(device, graph, &p, precision, 1);
        let per_sweep = five.kernel_ns.checked_sub(one.kernel_ns).unwrap() / 4;
        differences.push(per_sweep);
        eprintln!(
            "dense {precision:?} timing N={} one_sweep_ns={} five_sweeps_ns={} per_sweep_ns={per_sweep}",
            graph.num_nodes(),
            one.kernel_ns,
            five.kernel_ns
        );
    }
    differences.sort_unstable();
    eprintln!(
        "dense {precision:?} timing N={} replicas=128 instances=1 median_per_sweep_ns={}",
        graph.num_nodes(),
        differences[1]
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn dense_oversized_model_refused_before_allocation() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 2_000).unwrap();
    let limits = DeviceLimits {
        free_bytes: 64 << 20,
        ..device.limits
    };
    let device = device.with_limits_for_test(limits);
    let graph = circulant(2_000, 60, Precision::Binary);
    let p = SampleParams {
        num_reads: 128,
        num_sweeps: 1,
        ..SampleParams::default()
    };
    let ctx = CudaContext::new(0).unwrap();
    ctx.synchronize().unwrap();
    let (before, _) = ctx.mem_get_info().unwrap();
    let error =
        quip_solver_cuda::streaming::sample_one_real_forced(&device, &graph, &p, MsaStorage::Dense)
            .unwrap_err();
    let SampleError::ModelExceedsDeviceMemory {
        nodes,
        couplers,
        required_bytes,
        available_bytes,
        instances,
    } = error
    else {
        panic!("unexpected refusal: {error}");
    };
    assert_eq!(
        (nodes, couplers, instances),
        (2_000, 60_000, device.max_sms)
    );
    assert!(required_bytes > available_bytes);
    assert_eq!(available_bytes, limits.usable_bytes());
    ctx.synchronize().unwrap();
    let (after, _) = ctx.mem_get_info().unwrap();
    assert!(before.saturating_sub(after) <= 2 << 20);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn degree_62_sparse_63_dense() {
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 128).unwrap();
    let p = SampleParams {
        num_reads: 8,
        num_sweeps: 64,
        ..SampleParams::default()
    };
    let graph = circulant(128, 62, Precision::Binary);
    sample_ising(&device, &graph, &p, KernelKind::Msa).unwrap();
    assert_eq!(
        quip_solver_cuda::streaming::last_storage(&device),
        Some(MsaStorage::Sparse)
    );
    let mut graph = graph;
    let mut edges = graph.edges.to_vec();
    edges.push((0, 64));
    graph.edges = edges.into();
    graph.j.push(-1.0);
    sample_ising(&device, &graph, &p, KernelKind::Msa).unwrap();
    assert_eq!(
        quip_solver_cuda::streaming::last_storage(&device),
        Some(MsaStorage::Dense)
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn probe_capacity_json_parses() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_quip-cuda-msa"))
        .args(["--probe-capacity", "--json", "--device", "0"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["cells"].as_array().unwrap().len(), 9);
    assert_eq!(value["betas"], 64);
    assert!(value["reads"] == 128 || value["reads"] == 64);
    for cell in value["cells"].as_array().unwrap() {
        assert!(["shared", "spill", "dense"].contains(&cell["tier"].as_str().unwrap()));
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn probe_limit_runs_and_one_above_refuses() {
    use quip_solver_cuda::capacity::{msa_max_model, Connectivity};
    use quip_solver_cuda::streaming::sample_one_real;
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 5000).unwrap();
    let limits = DeviceLimits {
        free_bytes: 64 << 20,
        ..device.limits
    };
    let device = device.with_limits_for_test(limits);
    let p = SampleParams {
        num_reads: 128,
        num_sweeps: 64,
        sweeps_per_beta: 1,
        seed: 19,
        ..SampleParams::default()
    };
    for conn in [Connectivity::Degree(20), Connectivity::Complete] {
        for precision in [Precision::Binary, Precision::Float32] {
            let shape = msa_max_model(conn, precision, p.num_reads, &limits);
            assert!(shape.nodes > 63);
            let graph = |n| match conn {
                Connectivity::Degree(d) => circulant(n, d, precision),
                Connectivity::Complete => dense_graph(n, precision),
            };
            let reads = sample_one_real(&device, &graph(shape.nodes), &p).unwrap();
            assert_eq!(reads.len(), p.num_reads);
            assert!(
                matches!(
                    sample_one_real(&device, &graph(shape.nodes + 1), &p),
                    Err(SampleError::ModelExceedsDeviceMemory { .. })
                ),
                "{conn:?} {precision:?}"
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn shared_spin_cap_is_counted_before_allocation() {
    use quip_solver_cuda::capacity::{msa_max_model, Connectivity};
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 5000).unwrap();
    let limits = DeviceLimits {
        free_bytes: 64 << 20,
        ..device.limits
    };
    let shape = msa_max_model(Connectivity::Degree(20), Precision::Binary, 128, &limits);
    let mut device = device.with_limits_for_test(limits);
    device.msa_shared_spins_cap = Some(0);
    let p = SampleParams {
        num_reads: 128,
        num_sweeps: 64,
        ..SampleParams::default()
    };
    assert!(matches!(
        sample_ising(
            &device,
            &circulant(shape.nodes, 20, Precision::Binary),
            &p,
            KernelKind::Msa
        ),
        Err(SampleError::ModelExceedsDeviceMemory { .. })
    ));
}

//! GPU allocation measurements for the MSA capacity estimate.

use cudarc::driver::{sys::CUdevice_attribute, CudaContext};
use quip_solver_core::{CancelToken, StreamJob, StreamOutcome};
use quip_solver_cuda::capacity::{msa_footprint, DeviceLimits, ModelShape, MsaStorage, Precision};
use quip_solver_cuda::coefficients;
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::nvml_gov::UtilGovernor;
use quip_solver_cuda::streaming::{run_stream, stream_width};
use quip_solver_cuda::{IsingGraph, KernelKind, SampleParams};

fn circulant(nodes: usize, degree: usize, precision: Precision) -> IsingGraph {
    let mut edges = Vec::with_capacity(nodes * degree / 2);
    for node in 0..nodes {
        for offset in 1..=degree / 2 {
            edges.push((node, (node + offset) % nodes));
        }
    }
    let j = (0..edges.len())
        .map(|edge| {
            if precision == Precision::Int32 && edge % 2 == 0 {
                1.001
            } else {
                -1.0
            }
        })
        .collect();
    IsingGraph::new(vec![0.0; nodes], j, edges)
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn footprint_matches_measured_allocation() {
    let nodes = 50_000;
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, nodes).expect("MSA device");
    // cudarc retains the same primary context on the session's device.
    let ctx = CudaContext::new(device.device_index).expect("session device context");
    let attribute = |key| usize::try_from(ctx.attribute(key).expect("device attribute")).unwrap();
    let gov = UtilGovernor::start(&device.pci_bus_id, 100, false);
    for degree in [20, 60] {
        for precision in [Precision::Binary, Precision::Int32] {
            let graph = circulant(nodes, degree, precision);
            let plan = coefficients::plan(KernelKind::Msa, &graph, device.max_nodes).unwrap();
            assert_eq!(
                Precision::from_plan(plan, precision == Precision::Binary),
                precision
            );
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
            let footprint = msa_footprint(
                ModelShape {
                    nodes,
                    couplers: graph.edges.len(),
                    max_degree: degree,
                },
                precision,
                MsaStorage::Sparse,
                128,
                &limits,
            );
            assert_eq!(footprint.instances, stream_width(&device, KernelKind::Msa));
            let (job_tx, job_rx) = tokio::sync::mpsc::channel(1);
            let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(1);
            job_tx
                .blocking_send(StreamJob {
                    job_id: vec![0],
                    graph,
                    params: SampleParams {
                        num_reads: 128,
                        num_sweeps: 1,
                        ..SampleParams::default()
                    },
                    watermark: None,
                })
                .unwrap();
            let measured = std::thread::scope(|scope| {
                // run_stream allocates stream_width nonces even for one active job.
                let driver = scope.spawn(|| {
                    run_stream(
                        &device,
                        KernelKind::Msa,
                        job_rx,
                        out_tx,
                        CancelToken::default(),
                        &gov,
                    );
                });
                let result = out_rx.blocking_recv().expect("completed model");
                let (free_after, _) = ctx.mem_get_info().expect("free with session alive");
                // Keeping the input open holds the session until this measurement.
                drop(job_tx);
                drop(out_rx);
                driver.join().expect("session teardown");
                assert!(
                    matches!(result.outcome, StreamOutcome::Completed(Ok(_))),
                    "session must complete successfully"
                );
                free_before
                    .checked_sub(free_after)
                    .expect("session consumes memory")
            });
            // Drain queued frees before the next free-memory measurement.
            ctx.synchronize().unwrap();
            eprintln!("degree={degree} precision={precision:?} instances={} measured={measured} footprint={}", footprint.instances, footprint.total());
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

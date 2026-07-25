//! Headless (no GPU) integration coverage of this crate's public library surface.
//!
//! Closes [quip-miner-cuda-gp2] part (a): CI-effective tests that exercise
//! `streaming`, `topology`, and identity constants without a CUDA device.
//!
//! `stream_width` requires a `&CudaDevice` and cannot run headless — deliberately
//! not covered here. Run GPU tests with `cargo test -- --ignored` on a GPU host.

use quip_miner_cuda::streaming::max_reads;
use quip_miner_cuda::topology::{fill_h_j, SelfFeedingTopology};
use quip_miner_cuda::{Algorithm, IsingGraph, CUDA_GIBBS_IDENTITY, CUDA_SA_IDENTITY};

/// Read cap advertised by the streaming driver (kernel block size for SA).
#[test]
fn max_reads_is_256_for_sa_and_gibbs() {
    assert_eq!(max_reads(Algorithm::Sa), 256);
    assert_eq!(max_reads(Algorithm::Gibbs), 256);
}

/// Identity `max_nodes` must mirror the kernel fixed-size limits they reject against.
#[test]
fn cuda_identities_mirror_kernel_node_limits() {
    assert_eq!(CUDA_SA_IDENTITY.backend, "cuda");
    assert_eq!(CUDA_SA_IDENTITY.algorithm, "sa");
    assert_eq!(CUDA_SA_IDENTITY.max_nodes, 5000);

    assert_eq!(CUDA_GIBBS_IDENTITY.backend, "cuda");
    assert_eq!(CUDA_GIBBS_IDENTITY.algorithm, "gibbs");
    assert_eq!(CUDA_GIBBS_IDENTITY.max_nodes, 4800);
}

/// Exercise `SelfFeedingTopology::build` + `fill_h_j` via the public API only.
///
/// Topology struct fields may be `pub(crate)`; assertions use `fill_h_j` return
/// values (and their lengths, which encode `n` / `nnz`) rather than field access.
#[test]
fn topology_build_and_fill_h_j_on_small_ring() {
    // 4-node ring, consensus-range h/J (lossless int8 quantize).
    let graph = IsingGraph::new(
        vec![1.0, -1.0, 0.0, 1.0],
        vec![1.0, -1.0, 1.0, -1.0],
        vec![(0, 1), (1, 2), (2, 3), (3, 0)],
    );
    let topo = SelfFeedingTopology::build(&graph);
    let (j_csr, h_i8) = fill_h_j(&topo, &graph);

    assert_eq!(h_i8, vec![1i8, -1, 0, 1]);
    assert_eq!(h_i8.len(), 4, "h length tracks node count");
    // 4 undirected edges → 8 directed CSR halves, each non-zero for |J|=1.
    assert_eq!(j_csr.len(), 8, "CSR nnz for a 4-edge undirected ring");
    assert_eq!(
        j_csr.iter().filter(|&&v| v != 0).count(),
        8,
        "every directed half carries a quantized J"
    );
    // Multiset of |J| values: each undirected edge contributes two ±1 entries.
    let positives = j_csr.iter().filter(|&&v| v == 1).count();
    let negatives = j_csr.iter().filter(|&&v| v == -1).count();
    assert_eq!(positives, 4);
    assert_eq!(negatives, 4);
}

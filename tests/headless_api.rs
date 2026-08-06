//! Headless (no GPU) integration coverage of this crate's public library surface.
//!
//! Closes [quip-miner-cuda-gp2] part (a): CI-effective tests that exercise
//! `streaming`, `topology`, and identity constants without a CUDA device.
//!
//! `stream_width` requires a `&CudaDevice` and cannot run headless — deliberately
//! not covered here. Run GPU tests with `cargo test -- --ignored` on a GPU host.

use quip_miner_cuda::capacity::{GIBBS_DEFAULT_NODES, SA_DEFAULT_NODES};
use quip_miner_cuda::streaming::max_reads;
use quip_miner_cuda::topology::{fill_h_j, SelfFeedingTopology};
use quip_miner_cuda::{cuda_gibbs_identity, cuda_sa_identity, Algorithm, IsingGraph};

/// Read cap advertised by the streaming driver (kernel block size for SA).
#[test]
fn max_reads_is_256_for_sa_and_gibbs() {
    assert_eq!(max_reads(Algorithm::Sa), 256);
    assert_eq!(max_reads(Algorithm::Gibbs), 256);
}

/// Identity `max_nodes` must mirror whatever capacity the process resolved,
/// not a fixed constant, so `--capabilities` never overstates or understates
/// what the compiled kernel accepts.
#[test]
fn identities_report_the_resolved_capacity() {
    let sa = cuda_sa_identity(SA_DEFAULT_NODES);
    assert_eq!(sa.backend, "cuda");
    assert_eq!(sa.algorithm, "sa");
    assert_eq!(sa.max_nodes, 5000);

    let gibbs = cuda_gibbs_identity(GIBBS_DEFAULT_NODES);
    assert_eq!(gibbs.backend, "cuda");
    assert_eq!(gibbs.algorithm, "gibbs");
    assert_eq!(gibbs.max_nodes, 4800);

    // A raised capacity must show through, or a coordinator would keep
    // rejecting jobs the kernel can now accept.
    assert_eq!(cuda_sa_identity(8192).max_nodes, 8192);
    assert_eq!(cuda_gibbs_identity(32768).max_nodes, 32768);
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

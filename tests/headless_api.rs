//! Headless (no GPU) integration coverage of this crate's public library surface.
//!
//! Closes [quip-miner-cuda-gp2] part (a): CI-effective tests that exercise
//! `streaming`, `topology`, and identity constants without a CUDA device.
//!
//! `stream_width` requires a `&CudaDevice` and cannot run headless — deliberately
//! not covered here. Run GPU tests with `cargo test -- --ignored` on a GPU host.

use quip_solver_core::quip_protocol::session::{algorithm_name, backend_name};
use quip_solver_cuda::capacity::{
    GIBBS_DEFAULT_NODES, MSA_DEFAULT_NODES, MSA_MAX_READS, SA_DEFAULT_NODES,
};
use quip_solver_cuda::streaming::max_reads;
use quip_solver_cuda::topology::{fill_coefficients, SelfFeedingTopology};
use quip_solver_cuda::{
    cuda_gibbs_identity, cuda_msa_identity, cuda_sa_identity, IsingGraph, KernelKind,
};

/// Read cap advertised by the streaming driver (kernel block size for SA).
#[test]
fn max_reads_is_the_kernel_read_cap() {
    assert_eq!(max_reads(KernelKind::Sa), 256);
    assert_eq!(max_reads(KernelKind::Gibbs), 256);
    assert_eq!(max_reads(KernelKind::Msa), 128);
}

/// Identity `max_nodes` must mirror whatever capacity the process resolved,
/// not a fixed constant, so `--capabilities` never overstates or understates
/// what the compiled kernel accepts.
#[test]
fn identities_report_the_resolved_capacity() {
    let sa = cuda_sa_identity(SA_DEFAULT_NODES);
    assert_eq!(backend_name(sa.backend), "cuda");
    assert_eq!(algorithm_name(sa.algorithm), "sa");
    assert_eq!(sa.max_nodes, 5000);

    let gibbs = cuda_gibbs_identity(GIBBS_DEFAULT_NODES);
    assert_eq!(backend_name(gibbs.backend), "cuda");
    assert_eq!(algorithm_name(gibbs.algorithm), "gibbs");
    assert_eq!(gibbs.max_nodes, 4800);

    // A raised capacity must show through, or a coordinator would keep
    // rejecting jobs the kernel can now accept.
    assert_eq!(cuda_sa_identity(8192).max_nodes, 8192);
    assert_eq!(cuda_gibbs_identity(32768).max_nodes, 32768);
}

/// The msa identity pins reads at the kernel's cap; the two must not drift.
/// `min_reads == max_reads` is load-bearing: the adapt controller scales reads
/// with difficulty whenever they differ, and msa wants a fixed count.
#[test]
fn msa_identity_reads_match_the_kernel_read_cap() {
    let msa = cuda_msa_identity(MSA_DEFAULT_NODES, 123_456);
    assert_eq!(algorithm_name(msa.algorithm), "msa");
    assert_eq!(msa.max_nodes, 5000);
    assert_eq!(msa.adapt.min_reads, msa.adapt.max_reads);
    assert_eq!(
        usize::try_from(msa.adapt.max_reads).expect("small"),
        MSA_MAX_READS
    );
    assert_eq!(max_reads(KernelKind::Msa), msa.adapt.max_reads);
}

#[test]
fn msa_identity_declares_the_resolved_limits() {
    let id = quip_solver_cuda::cuda_msa_identity(1_234_567, 7_654_321);
    assert_eq!(id.max_nodes, 1_234_567);
    assert_eq!(id.max_edges, 7_654_321);
    assert_eq!(id.adapt.max_reads, 128);
    assert_eq!(id.limits.max_degree, Some(62));
}

/// Exercise `SelfFeedingTopology::build` + `fill_coefficients` via the public API only.
///
/// Topology struct fields may be `pub(crate)`; assertions use `fill_coefficients` return
/// values (and their lengths, which encode `n` / `nnz`) rather than field access.
#[test]
fn topology_build_and_fill_coefficients_on_small_ring() {
    // 4-node ring, consensus-range h/J (exact int8 coefficients).
    let graph = IsingGraph::new(
        vec![1.0, -1.0, 0.0, 1.0],
        vec![1.0, -1.0, 1.0, -1.0],
        vec![(0, 1), (1, 2), (2, 3), (3, 0)],
    );
    let topo = SelfFeedingTopology::build(&graph);
    let (j_csr, h_i8) = fill_coefficients::<i8>(&topo, &graph, 1000).unwrap();

    assert_eq!(h_i8, vec![1i8, -1, 0, 1]);
    assert_eq!(h_i8.len(), 4, "h length tracks node count");
    // 4 undirected edges → 8 directed CSR halves, each non-zero for |J|=1.
    assert_eq!(j_csr.len(), 8, "CSR nnz for a 4-edge undirected ring");
    assert_eq!(
        j_csr.iter().filter(|&&v| v != 0).count(),
        8,
        "every directed half carries an exact J"
    );
    // Multiset of |J| values: each undirected edge contributes two ±1 entries.
    let positives = j_csr.iter().filter(|&&v| v == 1).count();
    let negatives = j_csr.iter().filter(|&&v| v == -1).count();
    assert_eq!(positives, 4);
    assert_eq!(negatives, 4);
}

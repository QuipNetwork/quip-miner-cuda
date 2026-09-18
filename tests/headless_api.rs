//! Headless (no GPU) integration coverage of this crate's public library surface.
//!
//! Closes [quip-miner-cuda-gp2] part (a): CI-effective tests that exercise
//! `streaming`, `topology`, and identity constants without a CUDA device.
//!
//! `stream_width` requires a `&CudaDevice` and cannot run headless — deliberately
//! not covered here. Run GPU tests with `cargo test -- --ignored` on a GPU host.

use quip_miner_cuda::capacity::{
    GIBBS_DEFAULT_NODES, MSA_DEFAULT_NODES, MSA_MAX_READS, MSA_REPLICA_WORDS, SA_DEFAULT_NODES,
};
use quip_miner_cuda::streaming::max_reads;
use quip_miner_cuda::topology::{fill_h_j, SelfFeedingTopology};
use quip_miner_cuda::{
    cuda_gibbs_identity, cuda_msa_identity, cuda_sa_identity, IsingGraph, KernelKind,
};

/// Read cap advertised by the streaming driver (kernel block size for SA).
/// msa scales with the replica words its device holds; SA and Gibbs do not.
#[test]
fn max_reads_is_the_kernel_read_cap() {
    assert_eq!(max_reads(KernelKind::Sa, MSA_REPLICA_WORDS), 256);
    assert_eq!(max_reads(KernelKind::Gibbs, MSA_REPLICA_WORDS), 256);
    assert_eq!(max_reads(KernelKind::Msa, MSA_REPLICA_WORDS), 128);
    assert_eq!(max_reads(KernelKind::Sa, 1), 256);
    assert_eq!(max_reads(KernelKind::Gibbs, 1), 256);
    assert_eq!(max_reads(KernelKind::Msa, 1), 64);
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

/// The msa identity pins reads at the kernel's cap; the two must not drift.
/// `min_reads == max_reads` is load-bearing: the adapt controller scales reads
/// with difficulty whenever they differ, and msa wants a fixed count.
#[test]
fn msa_identity_reads_match_the_kernel_read_cap() {
    let msa = cuda_msa_identity(MSA_DEFAULT_NODES, 128);
    assert_eq!(msa.algorithm, "msa");
    assert_eq!(msa.max_nodes, 5000);
    assert_eq!(msa.adapt.min_reads, msa.adapt.max_reads);
    assert_eq!(
        usize::try_from(msa.adapt.max_reads).expect("small"),
        MSA_MAX_READS
    );
    assert_eq!(
        max_reads(KernelKind::Msa, MSA_REPLICA_WORDS),
        msa.adapt.max_reads
    );
}

/// A device that only holds one replica word declares 64 reads, still pinned.
/// Declaring 128 there would take 128-read jobs the session would clamp to 64,
/// reporting work it did not do (quip-miner-cuda-9p5).
#[test]
fn msa_identity_declares_the_one_word_read_cap() {
    let turing = cuda_msa_identity(MSA_DEFAULT_NODES, 64);
    assert_eq!(turing.adapt.min_reads, 64);
    assert_eq!(turing.adapt.max_reads, 64);
    assert_eq!(max_reads(KernelKind::Msa, 1), turing.adapt.max_reads);
    // The sweep envelope is unchanged by the word count.
    let full = cuda_msa_identity(MSA_DEFAULT_NODES, 128);
    assert_eq!(turing.adapt.min_sweeps, full.adapt.min_sweeps);
    assert_eq!(turing.adapt.max_sweeps, full.adapt.max_sweeps);
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

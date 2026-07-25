//! CSR topology + chromatic color-blocks + int8 quantization for the
//! self-feeding kernels.
//!
//! Mirrors `GPU/sampler_utils.py::build_csr_structure_from_edges` /
//! `build_edge_position_index` / `compute_color_blocks`, but computes a
//! generic greedy coloring instead of the Zephyr-specific linear-index
//! formula (`zephyr_four_color_linear`): the kernel takes `num_colors` as a
//! runtime argument, so any valid coloring (same-color nodes non-adjacent)
//! is correct, and a greedy coloring works for any topology, not just
//! Zephyr.
//!
//! Consensus `h`/`J` are constrained to small integers by protocol design
//! (`DEFAULT_ALLOWED_H = {-1,0,1}`, `DEFAULT_ALLOWED_J = {-1,1}`, milli
//! units; see `shared/quantum_proof_of_work.py`), so the int8 cast the
//! original kernel relies on is lossless for real jobs.

use quip_miner_core::IsingGraph;

/// Chromatic color-block partition of a CSR graph's dense node indices.
///
/// `nodes` is grouped by color; `starts`/`counts` index into it per color.
/// Same-color nodes are pairwise non-adjacent (independent set), which is
/// all the kernel's per-color parallel update requires.
///
/// Field layout (crate-private; same-crate code may read them directly):
/// - color `c` owns `nodes[starts[c] as usize .. (starts[c] + counts[c]) as usize]`
/// - `starts.len() == counts.len() == num_colors as usize`
/// - `counts.iter().sum::<i32>() == nodes.len() as i32` (equals `n`)
///
/// # Examples
///
/// External callers only observe coloring through the public topology +
/// `fill_h_j` surface; the partition fields are `pub(crate)`.
///
/// ```
/// use quip_miner_cuda::{IsingGraph, topology::{SelfFeedingTopology, fill_h_j}};
///
/// let graph = IsingGraph::new(
///     vec![0.0, 0.0, 0.0, 0.0],
///     vec![1.0, 1.0, 1.0, 1.0],
///     vec![(0, 1), (1, 2), (2, 3), (3, 0)],
/// );
/// let topology = SelfFeedingTopology::build(&graph);
/// // Proper coloring is an internal invariant of `build`; public surface is
/// // the quantized CSR layout produced by `fill_h_j`.
/// let (j_csr, h_i8) = fill_h_j(&topology, &graph);
/// assert_eq!(h_i8, vec![0, 0, 0, 0]);
/// assert_eq!(j_csr.len(), 8); // 4 undirected edges × 2 directed halves
/// assert!(j_csr.iter().all(|&v| v == 1));
/// ```
#[derive(Clone, Debug)]
pub struct ColorBlocks {
    pub(crate) starts: Vec<i32>,
    pub(crate) counts: Vec<i32>,
    pub(crate) nodes: Vec<i32>,
    pub(crate) num_colors: i32,
}

/// Greedy (Welsh-Powell) coloring of a CSR adjacency: process nodes in
/// degree-descending order, assign the smallest color unused by any
/// already-colored neighbor. Not the Zephyr-optimal 4-coloring, but valid
/// for any graph and typically close to it for sparse Ising topologies.
///
/// Working types stay `usize` for the traversal; convert to `i32` only when
/// packing `ColorBlocks` for the kernel ABI. CSR `row_ptr`/`col_ind` values
/// are non-negative and in-range by construction of [`SelfFeedingTopology::build`]
/// (out-of-range endpoints are skipped; self-loops and sorted adj stay ≥ 0).
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn greedy_color(n: usize, row_ptr: &[i32], col_ind: &[i32]) -> ColorBlocks {
    if n == 0 {
        return ColorBlocks {
            starts: Vec::new(),
            counts: Vec::new(),
            nodes: Vec::new(),
            num_colors: 0,
        };
    }
    // CSR i32 → usize for host indexing; values non-negative by construction.
    let degree = |i: usize| (row_ptr[i + 1] - row_ptr[i]) as usize;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(degree(i)));

    // `usize::MAX` = uncolored; colors otherwise in `0..n`.
    let mut color_of = vec![usize::MAX; n];
    let mut used = vec![false; n]; // reused scratch, cleared per node
    for &node in &order {
        let start = row_ptr[node] as usize;
        let end = row_ptr[node + 1] as usize;
        let mut touched: Vec<usize> = Vec::with_capacity(end - start);
        for &nbr in &col_ind[start..end] {
            // nbr is a node id written by build; always in 0..n.
            let c = color_of[nbr as usize];
            if c != usize::MAX {
                used[c] = true;
                touched.push(c);
            }
        }
        let mut c = 0usize;
        while c < n && used[c] {
            c += 1;
        }
        color_of[node] = c;
        for t in touched {
            used[t] = false;
        }
    }

    let num_colors = color_of
        .iter()
        .copied()
        .filter(|&c| c != usize::MAX)
        .max()
        .map_or(0, |m| m + 1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); num_colors];
    for (i, &c) in color_of.iter().enumerate() {
        if c != usize::MAX {
            groups[c].push(i);
        }
    }
    // Pack ColorBlocks: host usize → kernel i32 only at this boundary.
    let mut starts = Vec::with_capacity(groups.len());
    let mut counts = Vec::with_capacity(groups.len());
    let mut nodes = Vec::with_capacity(n);
    let mut cur = 0i32;
    for g in &groups {
        starts.push(cur);
        counts.push(g.len() as i32);
        nodes.extend(g.iter().map(|&i| i as i32));
        cur += g.len() as i32;
    }
    ColorBlocks {
        starts,
        counts,
        nodes,
        num_colors: num_colors as i32,
    }
}

/// Fixed CSR topology shared by every nonce/slot in a self-feeding session.
///
/// Built once from the first job's graph. Subsequent jobs must supply the
/// exact same `(n, edges)` (checked by the caller via [`IsingGraph`]
/// equality) to reuse it; `edge_pos` gives each edge's two CSR positions in
/// that fixed order, so per-job `J` upload is a direct scatter with no
/// per-job graph traversal.
///
/// Fields are `pub(crate)`: CSR layout, coloring, and `edge_pos` parallel to
/// the establishing edge order are invariant-bearing and only consumed inside
/// this crate (`streaming`, `topology` tests).
#[derive(Clone, Debug)]
pub struct SelfFeedingTopology {
    pub(crate) n: usize,
    pub(crate) nnz: usize,
    pub(crate) row_ptr: Vec<i32>,
    pub(crate) col_ind: Vec<i32>,
    /// Per-edge `(pos_ij, pos_ji)` into `col_ind`/`j` arrays, parallel to the
    /// establishing graph's `edges` order.
    pub(crate) edge_pos: Vec<(u32, u32)>,
    pub(crate) colors: ColorBlocks,
}

impl SelfFeedingTopology {
    /// Build CSR + coloring from a graph. `graph.edges` fixes the canonical
    /// edge order used by `edge_pos` (and thus by [`fill_h_j`] for this and
    /// every later job sharing this topology).
    ///
    /// Skips edges whose endpoints are out of `0..n` (`n = graph.h.len()`).
    /// Self-loops contribute a single directed CSR half. Returns an empty
    /// topology when `n == 0`. Infallible: no [`Result`], so no `# Errors`.
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_miner_cuda::{IsingGraph, topology::{SelfFeedingTopology, fill_h_j}};
    ///
    /// let graph = IsingGraph::new(
    ///     vec![1.0, -1.0, 0.0],
    ///     vec![2.0, -3.0],
    ///     vec![(0, 1), (1, 2)],
    /// );
    /// let topology = SelfFeedingTopology::build(&graph);
    /// // Fields are crate-private; the public check is fill_h_j output shape.
    /// let (j_csr, h_i8) = fill_h_j(&topology, &graph);
    /// assert_eq!(h_i8, vec![1, -1, 0]);
    /// assert_eq!(j_csr.len(), 4); // two undirected edges, both directions
    /// assert_eq!(j_csr.iter().filter(|&&v| v == 2).count(), 2);
    /// assert_eq!(j_csr.iter().filter(|&&v| v == -3).count(), 2);
    /// ```
    ///
    /// Host-side adjacency is built entirely in `usize`. Casts to `i32`/`u32`
    /// happen only when packing the CUDA kernel ABI arrays (`row_ptr`,
    /// `col_ind`, `edge_pos`); graph size is bounded well below `i32::MAX` by
    /// session limits (`DEFAULT_MAX_EDGES` / `max_nodes`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn build(graph: &IsingGraph) -> Self {
        let n = graph.h.len();
        // Per-node list of (neighbor, edge_index, is_forward_half): carries
        // the originating `graph.edges[edge_index]` through the sort so the
        // final CSR position can be written straight into `edge_pos` below,
        // with no post-hoc search for "where did this edge end up".
        let mut adj: Vec<Vec<(usize, usize, bool)>> = vec![Vec::new(); n];
        for (k, &(u, v)) in graph.edges.iter().enumerate() {
            if u >= n || v >= n {
                continue;
            }
            adj[u].push((v, k, true));
            if u != v {
                adj[v].push((u, k, false));
            }
        }
        for nbrs in &mut adj {
            nbrs.sort_unstable_by_key(|&(nbr, _, _)| nbr);
        }

        let mut row_ptr = vec![0i32; n + 1];
        let mut col_ind = Vec::new();
        // (0, 0) for an edge with an out-of-range endpoint: never read,
        // since `fill_h_j` skips those edges too (matches the guard above).
        let mut edge_pos = vec![(0u32, 0u32); graph.edges.len()];
        for i in 0..n {
            // Pack kernel ABI: host length/index → i32/u32 at write boundary.
            row_ptr[i] = col_ind.len() as i32;
            for &(nbr, k, is_forward) in &adj[i] {
                let pos = col_ind.len() as u32;
                col_ind.push(nbr as i32);
                if is_forward {
                    edge_pos[k].0 = pos;
                } else {
                    edge_pos[k].1 = pos;
                }
            }
        }
        row_ptr[n] = col_ind.len() as i32;
        let nnz = col_ind.len();

        let colors = greedy_color(n, &row_ptr, &col_ind);

        Self {
            n,
            nnz,
            row_ptr,
            col_ind,
            edge_pos,
            colors,
        }
    }
}

/// Truncating cast to int8, saturating on overflow (Rust's `as` semantics
/// since 1.45). Matches numpy's `dtype=np.int8` cast for the in-range
/// values consensus actually produces (h in {-1,0,1}, J in {-1,1}, milli
/// units); saturates instead of wrapping for out-of-range test fixtures.
#[allow(clippy::cast_possible_truncation)] // intentional f64→i8 quantize
fn quantize_i8(v: f64) -> i8 {
    v as i8
}

/// Quantize one job's `h`/`J` into the topology's fixed CSR layout.
///
/// `j_csr` has length `topology.nnz`; `h_i8` has length `graph.h.len()`.
/// Positions not touched by any edge stay `0` (matches `j_csr` being
/// allocated/cleared before this call).
///
/// Missing or short `graph.edges` / `graph.j` relative to
/// `topology.edge_pos` are handled fallibly: out-of-range edge indices are
/// skipped, and missing `j[k]` defaults to `0.0` (same style as the
/// short-`j` path). Self-loops write only the forward CSR half.
///
/// # Panics
///
/// This function does **not** panic when `graph.edges.len() <
/// topology.edge_pos.len()`: those slots are skipped. It also does not
/// panic when `graph.j` is shorter than the edge list (defaults to `0.0`).
/// Correct placement of `J` still requires that `topology` was built from
/// an establishing graph whose `(n, edges)` match the caller's graph
/// (checked by the session layer); mismatched topologies can still produce
/// incorrect values, but not an edges-index panic.
///
/// # Examples
///
/// ```
/// use quip_miner_cuda::{IsingGraph, topology::{SelfFeedingTopology, fill_h_j}};
///
/// let graph = IsingGraph::new(
///     vec![1.0, -1.0],
///     vec![5.0],
///     vec![(0, 1)],
/// );
/// let topology = SelfFeedingTopology::build(&graph);
/// let (j_csr, h_i8) = fill_h_j(&topology, &graph);
/// assert_eq!(h_i8, vec![1, -1]);
/// assert_eq!(j_csr, vec![5, 5]); // both directed halves
/// ```
#[must_use]
pub fn fill_h_j(topology: &SelfFeedingTopology, graph: &IsingGraph) -> (Vec<i8>, Vec<i8>) {
    let mut j_csr = vec![0i8; topology.nnz];
    for (k, &(pos_ij, pos_ji)) in topology.edge_pos.iter().enumerate() {
        // quip-miner-cuda-2sg: fallible get — edge_pos sized from establishing
        // graph; caller-supplied graph.edges may be shorter.
        let Some(&(u, v)) = graph.edges.get(k) else {
            continue;
        };
        if u >= topology.n || v >= topology.n {
            continue;
        }
        let val = quantize_i8(graph.j.get(k).copied().unwrap_or(0.0));
        j_csr[pos_ij as usize] = val;
        if u != v {
            j_csr[pos_ji as usize] = val;
        }
    }
    let h_i8: Vec<i8> = graph.h.iter().map(|&v| quantize_i8(v)).collect();
    (j_csr, h_i8)
}

#[cfg(test)]
mod tests {
    // Test fixtures use literal constants whose values are visible at the
    // call site; host↔kernel ABI casts in assertions are not production risk.
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    use super::*;
    use proptest::prelude::*;

    fn g() -> IsingGraph {
        // Small ring: 0-1-2-3-0, unit J, ternary h.
        IsingGraph::new(
            vec![1.0, -1.0, 0.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        )
    }

    #[test]
    fn csr_shape_and_symmetry() {
        let t = SelfFeedingTopology::build(&g());
        assert_eq!(t.n, 4);
        assert_eq!(t.nnz, 8); // 4 edges * 2 directed halves
        assert_eq!(t.row_ptr, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn coloring_is_proper() {
        let t = SelfFeedingTopology::build(&g());
        // Every node gets exactly one color; adjacent nodes differ.
        let mut color_of = vec![-1i32; t.n];
        for (c, (&start, &count)) in t.colors.starts.iter().zip(&t.colors.counts).enumerate() {
            for i in 0..count {
                let node = t.colors.nodes[(start + i) as usize] as usize;
                assert_eq!(color_of[node], -1, "node colored twice");
                color_of[node] = c as i32;
            }
        }
        assert!(color_of.iter().all(|&c| c >= 0), "every node colored");
        for i in 0..t.n {
            let s = t.row_ptr[i] as usize;
            let e = t.row_ptr[i + 1] as usize;
            for &nbr in &t.col_ind[s..e] {
                assert_ne!(
                    color_of[i], color_of[nbr as usize],
                    "adjacent nodes {i} and {nbr} share a color"
                );
            }
        }
    }

    #[test]
    fn quantization_is_lossless_for_consensus_range() {
        let t = SelfFeedingTopology::build(&g());
        let (j, h) = fill_h_j(&t, &g());
        assert_eq!(h, vec![1i8, -1, 0, 1]);
        // Each edge's J appears at both directed CSR positions.
        assert_eq!(j.iter().filter(|&&v| v != 0).count(), 8);
    }

    #[test]
    fn empty_graph_has_no_colors() {
        let t = SelfFeedingTopology::build(&IsingGraph::new(vec![], vec![], vec![]));
        assert_eq!(t.n, 0);
        assert_eq!(t.colors.num_colors, 0);
    }

    // --- quip-miner-cuda-60l: targeted unit coverage ---

    #[test]
    fn build_skips_out_of_range_endpoints() {
        // Edge (0,1) valid; (9,0) and (1,99) out of range for n=2.
        let graph = IsingGraph::new(
            vec![0.0, 0.0],
            vec![3.0, 4.0, 5.0],
            vec![(0, 1), (9, 0), (1, 99)],
        );
        let t = SelfFeedingTopology::build(&graph);
        assert_eq!(t.n, 2);
        assert_eq!(t.nnz, 2); // only the valid undirected edge → 2 halves
        assert_eq!(t.col_ind, vec![1, 0]);
        // OOR edges keep default (0,0) edge_pos and never enter CSR.
        assert_eq!(t.edge_pos[1], (0, 0));
        assert_eq!(t.edge_pos[2], (0, 0));
        let (j, _) = fill_h_j(&t, &graph);
        assert_eq!(j, vec![3, 3]);
    }

    #[test]
    fn build_and_fill_handle_self_loops() {
        let graph = IsingGraph::new(vec![1.0, 0.0], vec![7.0, 2.0], vec![(0, 0), (0, 1)]);
        let t = SelfFeedingTopology::build(&graph);
        // Self-loop: one directed half; undirected edge: two → nnz = 3.
        assert_eq!(t.nnz, 3);
        // Self-loop only sets the forward half of edge_pos.
        let (pos_ij, pos_ji) = t.edge_pos[0];
        assert_eq!(t.col_ind[pos_ij as usize], 0);
        assert_eq!(pos_ji, 0); // never written for self-loop
        let (j, h) = fill_h_j(&t, &graph);
        assert_eq!(h, vec![1, 0]);
        // Self-loop J appears once; (0,1) appears twice.
        assert_eq!(j.iter().filter(|&&v| v == 7).count(), 1);
        assert_eq!(j.iter().filter(|&&v| v == 2).count(), 2);
    }

    #[test]
    fn quantize_i8_saturates_via_fill_h_j() {
        // Doc claims saturation for out-of-range fixtures (Rust float→i8 as).
        let graph = IsingGraph::new(
            vec![300.0, -400.0, 0.0],
            vec![200.0, -300.0],
            vec![(0, 1), (1, 2)],
        );
        let t = SelfFeedingTopology::build(&graph);
        let (j, h) = fill_h_j(&t, &graph);
        assert_eq!(h, vec![i8::MAX, i8::MIN, 0]);
        assert!(j.contains(&i8::MAX));
        assert!(j.contains(&i8::MIN));
    }

    #[test]
    fn fill_h_j_short_j_defaults_to_zero() {
        // Two edges, j only for the first — second coupling defaults to 0.
        let graph = IsingGraph::new(vec![0.0, 0.0, 0.0], vec![9.0], vec![(0, 1), (1, 2)]);
        let t = SelfFeedingTopology::build(&graph);
        let (j, _) = fill_h_j(&t, &graph);
        assert_eq!(j.iter().filter(|&&v| v == 9).count(), 2);
        // Positions for the short-j edge stay 0.
        assert_eq!(j.iter().filter(|&&v| v == 0).count(), 2);
    }

    #[test]
    fn fill_h_j_skips_missing_edges_when_graph_shorter() {
        // Establishing topology has 2 edges; call with a shorter edge list.
        let establish = IsingGraph::new(vec![0.0, 0.0, 0.0], vec![1.0, 2.0], vec![(0, 1), (1, 2)]);
        let t = SelfFeedingTopology::build(&establish);
        let short = IsingGraph::new(vec![0.0, 0.0, 0.0], vec![5.0], vec![(0, 1)]);
        let (j, _) = fill_h_j(&t, &short);
        // Only the first edge's positions get 5; second edge_pos slot skipped.
        assert_eq!(j.iter().filter(|&&v| v == 5).count(), 2);
        assert_eq!(j.iter().filter(|&&v| v != 0).count(), 2);
    }

    #[test]
    fn edge_pos_places_j_at_both_csr_positions() {
        let graph = g();
        let t = SelfFeedingTopology::build(&graph);
        for (k, &(u, v)) in graph.edges.iter().enumerate() {
            let (pos_ij, pos_ji) = t.edge_pos[k];
            assert_eq!(t.col_ind[pos_ij as usize] as usize, v, "edge {k} forward");
            assert_eq!(t.col_ind[pos_ji as usize] as usize, u, "edge {k} reverse");
        }
        let (j, _) = fill_h_j(&t, &graph);
        for (k, &j_val) in graph.j.iter().enumerate() {
            let (pos_ij, pos_ji) = t.edge_pos[k];
            let q = quantize_i8(j_val);
            assert_eq!(j[pos_ij as usize], q);
            assert_eq!(j[pos_ji as usize], q);
        }
    }

    // --- quip-miner-cuda-um5: property tests ---

    /// Strategy: n nodes, arbitrary edges (incl. self-loops / OOR / dups),
    /// and `j` that may be shorter than `edges`.
    fn arb_ising() -> impl Strategy<Value = IsingGraph> {
        (0usize..=12).prop_flat_map(|n| {
            let edge_strat = prop::collection::vec(
                (
                    0usize..n.saturating_mul(2).max(1),
                    0usize..n.saturating_mul(2).max(1),
                ),
                0..=24,
            );
            let j_len = 0usize..=24;
            (
                Just(n),
                edge_strat,
                j_len,
                prop::collection::vec(-5.0f64..5.0, n),
            )
                .prop_flat_map(|(_n, edges, j_len, h)| {
                    let j_len = j_len.min(edges.len() + 4);
                    prop::collection::vec(-200.0f64..200.0, j_len)
                        .prop_map(move |j| IsingGraph::new(h.clone(), j, edges.clone()))
                })
        })
    }

    fn color_of(t: &SelfFeedingTopology) -> Vec<i32> {
        let mut color_of = vec![-1i32; t.n];
        for (c, (&start, &count)) in t.colors.starts.iter().zip(&t.colors.counts).enumerate() {
            for i in 0..count {
                let node = t.colors.nodes[(start + i) as usize] as usize;
                color_of[node] = c as i32;
            }
        }
        color_of
    }

    fn assert_partition(t: &SelfFeedingTopology) {
        let nc = t.colors.num_colors as usize;
        assert_eq!(t.colors.starts.len(), nc);
        assert_eq!(t.colors.counts.len(), nc);
        let sum: i32 = t.colors.counts.iter().sum();
        assert_eq!(sum as usize, t.n);
        assert_eq!(t.colors.nodes.len(), t.n);
        let mut seen = vec![false; t.n];
        for &node in &t.colors.nodes {
            let node = node as usize;
            assert!(node < t.n);
            assert!(!seen[node], "nodes not a permutation");
            seen[node] = true;
        }
        assert!(seen.iter().all(|&s| s) || t.n == 0);
        for c in 0..nc.saturating_sub(1) {
            assert_eq!(
                t.colors.starts[c] + t.colors.counts[c],
                t.colors.starts[c + 1]
            );
        }
        if nc > 0 {
            let last = nc - 1;
            assert_eq!(t.colors.starts[last] + t.colors.counts[last], t.n as i32);
        }
    }

    fn assert_csr(t: &SelfFeedingTopology) {
        assert_eq!(t.row_ptr.len(), t.n + 1);
        if t.n == 0 {
            assert_eq!(t.nnz, 0);
            assert!(t.col_ind.is_empty());
            return;
        }
        assert_eq!(t.row_ptr[0], 0);
        assert_eq!(t.row_ptr[t.n] as usize, t.col_ind.len());
        assert_eq!(t.col_ind.len(), t.nnz);
        for i in 0..t.n {
            assert!(t.row_ptr[i] <= t.row_ptr[i + 1]);
        }
        // Adjacency symmetry: every i→j entry has a matching j→i entry
        // (multiset equality of directed pairs).
        let mut directed: Vec<(i32, i32)> = Vec::with_capacity(t.nnz);
        for i in 0..t.n {
            let s = t.row_ptr[i] as usize;
            let e = t.row_ptr[i + 1] as usize;
            for &nbr in &t.col_ind[s..e] {
                directed.push((i as i32, nbr));
            }
        }
        let mut rev: Vec<(i32, i32)> = directed.iter().map(|&(a, b)| (b, a)).collect();
        directed.sort_unstable();
        rev.sort_unstable();
        assert_eq!(directed, rev, "CSR adjacency not symmetric");
    }

    fn assert_edge_pos(t: &SelfFeedingTopology, graph: &IsingGraph) {
        assert_eq!(t.edge_pos.len(), graph.edges.len());
        for (k, &(u, v)) in graph.edges.iter().enumerate() {
            if u >= t.n || v >= t.n {
                assert_eq!(t.edge_pos[k], (0, 0));
                continue;
            }
            let (pos_ij, pos_ji) = t.edge_pos[k];
            assert_eq!(t.col_ind[pos_ij as usize] as usize, v);
            if u != v {
                assert_eq!(t.col_ind[pos_ji as usize] as usize, u);
            }
        }
    }

    fn assert_fill_h_j(t: &SelfFeedingTopology, graph: &IsingGraph) {
        let (j_csr, h_i8) = fill_h_j(t, graph);
        assert_eq!(j_csr.len(), t.nnz);
        assert_eq!(h_i8.len(), graph.h.len());
        for (i, &hv) in graph.h.iter().enumerate() {
            assert_eq!(h_i8[i], quantize_i8(hv));
        }
        // Untouched positions stay 0: build a mask of written indices.
        let mut written = vec![false; t.nnz];
        for (k, &(pos_ij, pos_ji)) in t.edge_pos.iter().enumerate() {
            let Some(&(u, v)) = graph.edges.get(k) else {
                continue;
            };
            if u >= t.n || v >= t.n {
                continue;
            }
            let val = quantize_i8(graph.j.get(k).copied().unwrap_or(0.0));
            assert_eq!(j_csr[pos_ij as usize], val);
            written[pos_ij as usize] = true;
            if u != v {
                assert_eq!(j_csr[pos_ji as usize], val);
                written[pos_ji as usize] = true;
                // Symmetry of the two halves for this edge.
                assert_eq!(j_csr[pos_ij as usize], j_csr[pos_ji as usize]);
            }
        }
        for (i, &w) in written.iter().enumerate() {
            if !w {
                assert_eq!(j_csr[i], 0, "untouched j_csr[{i}] must stay 0");
            }
        }
    }

    proptest! {
        #[test]
        fn prop_topology_invariants(graph in arb_ising()) {
            let t = SelfFeedingTopology::build(&graph);
            assert_eq!(t.n, graph.h.len());

            // Proper coloring: adjacent nodes differ (self-loops are adjacent
            // to themselves — greedy still assigns a color; the kernel only
            // needs same-color nodes to be non-adjacent for distinct pairs.
            // A self-loop means the node is adjacent to itself, so a "proper"
            // coloring in the strict sense is impossible; skip self-adj check.
            let colors = color_of(&t);
            if t.n > 0 {
                assert_eq!(colors.len(), t.n);
                assert!(colors.iter().all(|&c| c >= 0));
            }
            for i in 0..t.n {
                let s = t.row_ptr[i] as usize;
                let e = t.row_ptr[i + 1] as usize;
                for &nbr in &t.col_ind[s..e] {
                    let j = nbr as usize;
                    if i == j {
                        continue; // self-loop: cannot differ from self
                    }
                    assert_ne!(
                        colors[i], colors[j],
                        "adjacent {i},{j} share color"
                    );
                }
            }

            assert_partition(&t);
            assert_csr(&t);
            assert_edge_pos(&t, &graph);
            assert_fill_h_j(&t, &graph);
        }
    }

    proptest! {
        #[test]
        fn prop_fill_h_j_with_shorter_edges(
            n in 1usize..=8,
            edge_count in 1usize..=12,
        ) {
            let edges: Vec<(usize, usize)> = (0..edge_count)
                .map(|k| (k % n, (k + 1) % n))
                .collect();
            let j: Vec<f64> = (0..edge_count).map(|k| (k as f64) + 1.0).collect();
            let h = vec![0.0; n];
            let establish = IsingGraph::new(h.clone(), j, edges.clone());
            let t = SelfFeedingTopology::build(&establish);
            // Shorter job graph: drop last half of edges / j.
            let keep = edge_count / 2;
            let short = IsingGraph::new(
                h,
                establish.j[..keep].to_vec(),
                edges[..keep].to_vec(),
            );
            assert_fill_h_j(&t, &short);
        }
    }
}

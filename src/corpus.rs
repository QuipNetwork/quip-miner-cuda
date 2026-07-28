//! Real-corpus loading for `bench run --source/--topology`.
//!
//! The synthetic ring problem (`bench::ring_graph`) has degree 2 and is not
//! timing-representative of the real corpus topology (thousands of nodes,
//! degree ~18). This module redraws the real `(h, J)` the coordinator would
//! have sent a miner for a given nonce, from the same two files the
//! coordinator↔miner protocol already agrees on:
//!
//! - `--topology <spec.json>`: `{nodes, edges, allowed_h_milli,
//!   allowed_j_milli}`, the same shape as the coordinator's
//!   `TopologySpecJson` (`quip-coordinator/src/drive/topology_spec.rs`,
//!   duplicated here since that struct is private to the coordinator crate).
//! - `--source <corpus.jsonl>`: one `InstanceRecord`-shaped line per model
//!   (`quip-coordinator/src/download/record.rs`), keyed on `nonce`; unknown
//!   fields are ignored.
//!
//! Redraw uses [`draw_ising_milli`] — the same golden-pinned draw the
//! network uses (`quip-coordinator/src/producer/pow.rs`) — then maps each
//! edge's native (possibly sparse) node ids to dense `0..n` positions via
//! `nodes`' order, matching `TopologyCache::from_proto` in `quip-miner-core`.

use quip_miner_core::IsingGraph;
use quip_protocol::chacha8::draw_ising_milli;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// A parsed `--topology <spec.json>` file.
#[derive(Debug, Deserialize)]
pub struct TopologySpec {
    /// Topology node ids (native, possibly sparse); order fixes `h`'s draw order.
    pub nodes: Vec<u32>,
    /// Undirected edges as native node-id pairs; order fixes `J`'s draw order.
    #[serde(default)]
    pub edges: Vec<(u32, u32)>,
    /// Allowed linear-field values (milli).
    pub allowed_h_milli: Vec<i32>,
    /// Allowed coupling values (milli).
    pub allowed_j_milli: Vec<i32>,
}

/// One `--source <corpus.jsonl>` line. Extra keys (`energy_milli`,
/// `salt_hex`, `qblock_id`, `difficulty`, `provenance`, ...) are ignored;
/// only `nonce` and `topology_hash` are read.
#[derive(Debug, Deserialize)]
pub struct CorpusRecord {
    /// `ChaCha8` seed as un-prefixed 64-char hex (32 bytes).
    pub nonce: String,
    /// Topology bucket hash, passed through into the emitted `BenchRecord`.
    #[serde(default)]
    pub topology_hash: String,
}

/// Errors loading/parsing a `--source`/`--topology` corpus bench input.
#[derive(Debug, Error)]
pub enum CorpusError {
    /// Filesystem read failure.
    #[error("corpus io: {0}")]
    Io(String),
    /// The topology spec JSON failed to decode or was structurally invalid.
    #[error("topology spec: {0}")]
    TopologySpec(String),
    /// `reason` names the line's failure (bad JSON, bad nonce hex, redraw
    /// failure, or an edge referencing a node id absent from `nodes`);
    /// `line` is 1-based.
    #[error("corpus line {line}: {reason}")]
    Record {
        /// 1-based line number in the JSONL corpus file.
        line: usize,
        /// Failure detail.
        reason: String,
    },
}

/// Decode a 64-char hex nonce into its 32 raw bytes.
fn hex_to_32(hex: &str) -> Result<[u8; 32], String> {
    let hex = hex.trim();
    if hex.len() != 64 {
        return Err(format!(
            "nonce must be un-prefixed 64-char hex (32 bytes), got {} chars",
            hex.len()
        ));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Parse a `--topology <spec.json>` document.
///
/// # Errors
/// [`CorpusError::TopologySpec`] if the JSON does not decode into the
/// expected shape.
pub fn parse_topology_spec(text: &str) -> Result<TopologySpec, CorpusError> {
    serde_json::from_str(text).map_err(|e| CorpusError::TopologySpec(e.to_string()))
}

/// Read and parse a `--topology <spec.json>` file.
///
/// # Errors
/// [`CorpusError::Io`] if the file cannot be read; [`CorpusError::TopologySpec`]
/// if the JSON does not decode into the expected shape.
pub fn load_topology_spec(path: &Path) -> Result<TopologySpec, CorpusError> {
    let text = std::fs::read_to_string(path).map_err(|e| CorpusError::Io(e.to_string()))?;
    parse_topology_spec(&text)
}

/// Redraw one nonce's `(h, J)` against `spec` and build the real
/// [`IsingGraph`] the coordinator would have sent a miner for a
/// `TopologyHash` job referencing this spec.
///
/// Dense node positions are the index of each native id in `spec.nodes`'
/// order — the same mapping `TopologyCache::from_proto` builds in
/// `quip-miner-core` when a miner resolves a `TopologyHash` job.
fn build_graph(spec: &TopologySpec, nonce: [u8; 32]) -> Result<IsingGraph, String> {
    let (h_milli, j_milli) = draw_ising_milli(
        nonce,
        spec.nodes.len(),
        spec.edges.len(),
        &spec.allowed_h_milli,
        &spec.allowed_j_milli,
    )
    .map_err(|e| e.to_string())?;

    let pos: HashMap<u32, usize> = spec
        .nodes
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut edges = Vec::with_capacity(spec.edges.len());
    for (k, &(u, v)) in spec.edges.iter().enumerate() {
        let pu = *pos
            .get(&u)
            .ok_or_else(|| format!("edge {k} references unknown node id {u}"))?;
        let pv = *pos
            .get(&v)
            .ok_or_else(|| format!("edge {k} references unknown node id {v}"))?;
        edges.push((pu, pv));
    }
    let h: Vec<f64> = h_milli.iter().map(|&v| f64::from(v) / 1000.0).collect();
    let j: Vec<f64> = j_milli.iter().map(|&v| f64::from(v) / 1000.0).collect();
    Ok(IsingGraph::new(h, j, edges))
}

/// Parse a `--source <corpus.jsonl>` corpus (already read into `text`),
/// redrawing each nonce's real graph against `spec`. Blank lines are
/// skipped; at most `limit` records are returned (all, if `None`).
///
/// # Errors
/// [`CorpusError::Record`] at the first line that fails to decode, whose
/// nonce is not valid 64-char hex, or whose redraw/edge-resolution fails.
pub fn parse_corpus(
    text: &str,
    spec: &TopologySpec,
    limit: Option<usize>,
) -> Result<Vec<(CorpusRecord, IsingGraph)>, CorpusError> {
    let mut out = Vec::new();
    for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        if limit.is_some_and(|n| out.len() >= n) {
            break;
        }
        let line_no = i + 1;
        let to_record_err = |reason: String| CorpusError::Record {
            line: line_no,
            reason,
        };
        let record: CorpusRecord =
            serde_json::from_str(line).map_err(|e| to_record_err(e.to_string()))?;
        let nonce = hex_to_32(&record.nonce).map_err(&to_record_err)?;
        let graph = build_graph(spec, nonce).map_err(&to_record_err)?;
        out.push((record, graph));
    }
    Ok(out)
}

/// Read and parse a `--source <corpus.jsonl>` file; see [`parse_corpus`].
///
/// # Errors
/// [`CorpusError::Io`] if the file cannot be read; see [`parse_corpus`] for
/// per-line failures.
pub fn load_corpus(
    path: &Path,
    spec: &TopologySpec,
    limit: Option<usize>,
) -> Result<Vec<(CorpusRecord, IsingGraph)>, CorpusError> {
    let text = std::fs::read_to_string(path).map_err(|e| CorpusError::Io(e.to_string()))?;
    parse_corpus(&text, spec, limit)
}

#[cfg(test)]
mod tests {
    use super::{parse_corpus, parse_topology_spec, CorpusError};

    const RING4: &str = r#"{
        "nodes": [0, 1, 2, 3],
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "allowed_h_milli": [-1000, 0, 1000],
        "allowed_j_milli": [-1000, 1000]
    }"#;
    const NONCE_HEX: &str = "b4179357b751254ed0e68b5e969dcb50e73fd8c56be192b79d286ff2722d6a72";

    #[test]
    fn parses_valid_topology_spec() {
        let spec = parse_topology_spec(RING4).unwrap();
        assert_eq!(spec.nodes, vec![0, 1, 2, 3]);
        assert_eq!(spec.edges.len(), 4);
        assert_eq!(spec.allowed_h_milli, vec![-1000, 0, 1000]);
    }

    #[test]
    fn rejects_malformed_topology_json() {
        assert!(matches!(
            parse_topology_spec("not json"),
            Err(CorpusError::TopologySpec(_))
        ));
    }

    /// Golden vector (`conformance/golden_vectors.json`'s `ising[0]`): the
    /// redrawn graph's `h`/`j`/edges must match the golden `h_milli`/`j_milli`
    /// (scaled) and edge list exactly, proving the corpus path calls
    /// `draw_ising_milli` with the right arguments and doesn't misalign `J`
    /// against `edges`.
    #[test]
    fn build_graph_matches_golden_vector_on_dense_ids() {
        let spec = parse_topology_spec(RING4).unwrap();
        let corpus = format!("{{\"nonce\":\"{NONCE_HEX}\"}}\n");
        let recs = parse_corpus(&corpus, &spec, None).unwrap();
        assert_eq!(recs.len(), 1);
        let (record, graph) = &recs[0];
        assert_eq!(record.nonce, NONCE_HEX);
        assert_eq!(record.topology_hash, "", "absent field defaults to empty");
        assert_eq!(graph.h, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(graph.j, vec![1.0, -1.0, -1.0, 1.0]);
        assert_eq!(graph.edges, vec![(0, 1), (1, 2), (2, 3), (0, 3)]);
    }

    #[test]
    fn corpus_record_ignores_unknown_fields_and_keeps_topology_hash() {
        let spec = parse_topology_spec(RING4).unwrap();
        let corpus = format!(
            "{{\"nonce\":\"{NONCE_HEX}\",\"topology_hash\":\"deadbeef\",\"energy_milli\":-42,\"extra\":{{\"a\":1}}}}\n"
        );
        let recs = parse_corpus(&corpus, &spec, None).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].0.topology_hash, "deadbeef");
    }

    #[test]
    fn native_node_ids_remap_to_dense_positions() {
        // Sparse ids [10, 20, 30]: h draws in that order; edges use ids, not
        // positions, and must remap to (0,1) and (1,2).
        let spec = parse_topology_spec(
            r#"{
                "nodes": [10, 20, 30],
                "edges": [[10, 20], [20, 30]],
                "allowed_h_milli": [0],
                "allowed_j_milli": [1000]
            }"#,
        )
        .unwrap();
        let corpus = format!("{{\"nonce\":\"{NONCE_HEX}\"}}\n");
        let recs = parse_corpus(&corpus, &spec, None).unwrap();
        let (_, graph) = &recs[0];
        assert_eq!(graph.h, vec![0.0, 0.0, 0.0]);
        assert_eq!(graph.j, vec![1.0, 1.0]);
        assert_eq!(graph.edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn unknown_edge_node_id_is_a_record_error() {
        let spec = parse_topology_spec(
            r#"{
                "nodes": [0, 1],
                "edges": [[0, 5]],
                "allowed_h_milli": [0],
                "allowed_j_milli": [1000]
            }"#,
        )
        .unwrap();
        let corpus = format!("{{\"nonce\":\"{NONCE_HEX}\"}}\n");
        let err = parse_corpus(&corpus, &spec, None).unwrap_err();
        match err {
            CorpusError::Record { line, reason } => {
                assert_eq!(line, 1);
                assert!(reason.contains("unknown node id 5"), "{reason}");
            }
            other => panic!("expected Record error, got {other:?}"),
        }
    }

    #[test]
    fn bad_nonce_hex_is_a_record_error_naming_the_line() {
        let spec = parse_topology_spec(RING4).unwrap();
        let corpus = "{\"nonce\":\"ok\"}\n";
        let err = parse_corpus(corpus, &spec, None).unwrap_err();
        assert!(matches!(err, CorpusError::Record { line: 1, .. }));
    }

    #[test]
    fn malformed_json_line_is_a_record_error() {
        let spec = parse_topology_spec(RING4).unwrap();
        let err = parse_corpus("not json\n", &spec, None).unwrap_err();
        assert!(matches!(err, CorpusError::Record { line: 1, .. }));
    }

    #[test]
    fn limit_truncates_and_blank_lines_are_skipped() {
        let spec = parse_topology_spec(RING4).unwrap();
        let corpus = format!(
            "\n{{\"nonce\":\"{NONCE_HEX}\"}}\n\n{{\"nonce\":\"{NONCE_HEX}\"}}\n{{\"nonce\":\"{NONCE_HEX}\"}}\n"
        );
        let all = parse_corpus(&corpus, &spec, None).unwrap();
        assert_eq!(all.len(), 3);
        let limited = parse_corpus(&corpus, &spec, Some(2)).unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[test]
    fn empty_corpus_yields_no_records() {
        let spec = parse_topology_spec(RING4).unwrap();
        assert!(parse_corpus("", &spec, None).unwrap().is_empty());
    }
}

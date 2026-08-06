//! Unified per-part timing schema shared with sub-project C's Python viz.
//!
//! One [`BenchRecord`] is emitted per `(model, backend, config)` bench run.
//! Field names and the `scope`/`source` string forms are the contract locked
//! in `~/.claude/plans/isingmark/05-reconciliation.md`; changing them breaks
//! the downstream `benchmarks.jsonl` ingestion. Percentage-valued Nsight
//! metrics (SM efficiency, achieved occupancy) live in [`BenchRecord::metrics`]
//! rather than `parts`, so they never pollute the nanosecond time budget that
//! `residual_ns` reconciles against.

use serde::{Deserialize, Serialize};

/// Whether a timed part ran on the host (CPU driver) or the device (GPU).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Host-side driver work (allocation, memcpy enqueue, scoring).
    Host,
    /// Device-side work (kernel execution, measured by event or Nsight).
    Device,
}

/// How a part's or metric's value was obtained. Distinguishes measured from
/// derived and host-instrumented from Nsight-parsed.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    /// `tracing` span duration (host seam).
    Tracing,
    /// Derived from a measured aggregate ÷ a counted frequency.
    CounterDerived,
    /// `CudaEvent` start/stop elapsed time.
    CudaEvent,
    /// Parsed from an `nsys` stats report.
    Nsys,
    /// Parsed from an `ncu --csv` report.
    Ncu,
}

/// One timed part of the annealing pipeline. `per_call_ns` is the headline
/// figure ("a single sweep takes X ns"); it is always `total_ns / count`.
///
/// `per_call_ns` is `f64`: derived per-spin/per-flip costs are sub-nanosecond
/// and a `u64` would truncate them to zero (locked in the cross-plan
/// reconciliation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Part {
    /// Part name, from the fixed enum in the C plan (`sweep`, `upload`, …).
    pub part: String,
    /// Host or device.
    pub scope: Scope,
    /// Nanoseconds summed over the run.
    pub total_ns: u64,
    /// How many times the part ran (its frequency).
    pub count: u64,
    /// `total_ns as f64 / count.max(1) as f64`.
    pub per_call_ns: f64,
    /// How the timing was obtained.
    pub source: Source,
}

impl Part {
    /// Build a part, computing `per_call_ns` from `total_ns` and `count`.
    #[must_use]
    pub fn new(part: &str, scope: Scope, total_ns: u64, count: u64, source: Source) -> Self {
        // count.max(1): a zero-frequency part (never observed) still reports
        // its (zero) total rather than dividing by zero.
        #[allow(clippy::cast_precision_loss)] // ns totals stay far below 2^53 for one bench run
        let per_call_ns = total_ns as f64 / count.max(1) as f64;
        Self {
            part: part.to_owned(),
            scope,
            total_ns,
            count,
            per_call_ns,
            source,
        }
    }
}

/// A non-time metric (percentage, ratio, …) attached to a bench run. Kept
/// separate from [`Part`] so occupancy/efficiency never enter the ns time
/// budget `residual_ns` reconciles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metric {
    /// Metric name, e.g. `sm_efficiency`, `achieved_occupancy`.
    pub name: String,
    /// The metric's value in `unit`.
    pub value: f64,
    /// Unit string, e.g. `%`.
    pub unit: String,
    /// How the metric was obtained (always `nsys`/`ncu` today).
    pub source: Source,
}

impl Metric {
    /// Build a metric.
    #[must_use]
    pub fn new(name: &str, value: f64, unit: &str, source: Source) -> Self {
        Self {
            name: name.to_owned(),
            value,
            unit: unit.to_owned(),
            source,
        }
    }
}

/// One bench run's full per-part record. Serializes to the C-plan JSON object.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchRecord {
    /// `cuda-sa` | `cuda-gibbs`.
    pub backend: String,
    /// GPU name (from the device attribute query).
    pub device: String,
    /// Hex nonce of the model, or a synthetic id for ad-hoc problems.
    pub nonce: String,
    /// Hex topology hash, or a synthetic id.
    pub topology_hash: String,
    /// Reads per model.
    pub reads: u64,
    /// `num_sweeps` requested.
    pub sweeps: u64,
    /// Sweeps per beta rung.
    pub sweeps_per_beta: u64,
    /// Node count `N`.
    pub nodes: u64,
    /// Edge count `E`.
    pub edges: u64,
    /// Measured whole-model host wall time (ns).
    pub model_total_ns: u64,
    /// Every timed part.
    pub parts: Vec<Part>,
    /// Non-time Nsight metrics (SM efficiency, achieved occupancy, …).
    /// Empty unless a `bench fold` step ran; omitted from JSON when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics: Vec<Metric>,
    /// `model_total_ns - Σ(host-part total_ns)`; a self-consistency check.
    pub residual_ns: u64,
}

impl BenchRecord {
    /// Set `residual_ns` from the current host-scope parts. Device parts are
    /// excluded: they overlap the host `launch` wall rather than adding to it.
    ///
    /// Saturates at zero rather than going negative: `residual_ns` is `u64`
    /// per the locked schema, so a host-part sum that (through measurement
    /// noise) exceeds `model_total_ns` reports "no slack left", not a wrapped
    /// underflow.
    pub fn finalize(&mut self) {
        let host_sum: u64 = self
            .parts
            .iter()
            .filter(|p| p.scope == Scope::Host)
            .map(|p| p.total_ns)
            .sum();
        self.residual_ns = self.model_total_ns.saturating_sub(host_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::{BenchRecord, Metric, Part, Scope, Source};

    #[test]
    fn part_computes_per_call_and_record_computes_host_residual() {
        let mut rec = BenchRecord {
            backend: "cuda-sa".into(),
            device: "TestGPU".into(),
            nonce: "0x01".into(),
            topology_hash: "0xaa".into(),
            reads: 8,
            sweeps: 1024,
            sweeps_per_beta: 4,
            nodes: 100,
            edges: 200,
            model_total_ns: 1_000_000,
            parts: vec![
                Part::new("upload", Scope::Host, 300_000, 2, Source::CudaEvent),
                Part::new("launch", Scope::Device, 700_000, 1, Source::Nsys),
                Part::new("energy_score", Scope::Host, 100_000, 8, Source::Tracing),
            ],
            metrics: vec![],
            residual_ns: 0,
        };
        rec.finalize();
        // per_call_ns = total_ns / count, as f64.
        assert!((rec.parts[0].per_call_ns - 150_000.0).abs() < f64::EPSILON);
        assert!((rec.parts[2].per_call_ns - 12_500.0).abs() < f64::EPSILON);
        // residual excludes device-scope parts (they overlap host `launch` wall).
        assert_eq!(rec.residual_ns, 1_000_000 - (300_000 + 100_000));
    }

    #[test]
    fn finalize_saturates_at_zero_when_host_sum_exceeds_model_total() {
        let mut rec = BenchRecord {
            backend: "cuda-sa".into(),
            device: "TestGPU".into(),
            nonce: "0x01".into(),
            topology_hash: "0xaa".into(),
            reads: 1,
            sweeps: 1,
            sweeps_per_beta: 1,
            nodes: 1,
            edges: 0,
            model_total_ns: 100,
            parts: vec![Part::new("upload", Scope::Host, 500, 1, Source::CudaEvent)],
            metrics: vec![],
            residual_ns: 0,
        };
        rec.finalize();
        assert_eq!(rec.residual_ns, 0);
    }

    #[test]
    fn source_and_scope_serialize_to_contract_strings() {
        assert_eq!(serde_json::to_string(&Scope::Device).unwrap(), "\"device\"");
        assert_eq!(
            serde_json::to_string(&Source::CounterDerived).unwrap(),
            "\"counter-derived\""
        );
    }

    #[test]
    fn per_call_ns_derives_sub_nanosecond_costs_without_truncating() {
        // 100 ns over 3 calls is not integral; f64 must preserve the fraction.
        let p = Part::new("flip", Scope::Host, 100, 3, Source::CounterDerived);
        assert!((p.per_call_ns - 100.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_omitted_from_json_when_empty_but_present_when_populated() {
        let mut rec = BenchRecord {
            backend: "cuda-sa".into(),
            device: "TestGPU".into(),
            nonce: "0x01".into(),
            topology_hash: "0xaa".into(),
            reads: 1,
            sweeps: 1,
            sweeps_per_beta: 1,
            nodes: 1,
            edges: 0,
            model_total_ns: 10,
            parts: vec![],
            metrics: vec![],
            residual_ns: 0,
        };
        rec.finalize();
        let json = serde_json::to_string(&rec).unwrap();
        assert!(!json.contains("metrics"), "empty metrics must be omitted");

        rec.metrics
            .push(Metric::new("sm_efficiency", 78.4, "%", Source::Ncu));
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("sm_efficiency"));
        assert!(json.contains("\"unit\":\"%\""));
    }
}

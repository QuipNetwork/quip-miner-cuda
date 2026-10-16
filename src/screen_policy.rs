//! Single-stage lease screening policy. GPU-share feedback is fixed at zero.

use crate::cutoff::{Cutoff, CutoffConfig};
use crate::screen::{ScreenParams, MAX_PROBE_SWEEPS};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Read;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScreenConfig {
    pub(crate) enabled: bool,
    pub(crate) keep_denominator: u32,
    pub(crate) audit_denominator: u32,
    pub(crate) probe_reads_log2: u32,
    pub(crate) probe_sweeps: usize,
}

impl Default for ScreenConfig {
    fn default() -> Self {
        // Calibration: docs/perf/2026-10-cuda-lease-screen.md.
        Self {
            enabled: true,
            keep_denominator: 16,
            audit_denominator: 200,
            probe_reads_log2: 2,
            probe_sweeps: 512,
        }
    }
}

#[derive(serde::Deserialize, Default)]
pub(crate) struct ScreenToml {
    enabled: Option<bool>,
    keep_denominator: Option<i64>,
    audit_denominator: Option<i64>,
    probe_reads_log2: Option<i64>,
    probe_sweeps: Option<i64>,
    #[serde(flatten)]
    pub(crate) unknown: BTreeMap<String, toml::Value>,
}

impl ScreenConfig {
    pub(crate) fn merge(&mut self, cfg: &ScreenToml) {
        if let Some(enabled) = cfg.enabled {
            self.enabled = enabled;
        }
        if let Some(value) = checked_value("keep_denominator", cfg.keep_denominator, 2..=100_000) {
            self.keep_denominator = value;
        }
        if let Some(value) =
            checked_value("audit_denominator", cfg.audit_denominator, 0..=1_000_000)
        {
            if value == 1 {
                tracing::error!(
                    value,
                    "invalid screen.audit_denominator; retaining previous value"
                );
            } else {
                self.audit_denominator = value;
            }
        }
        if let Some(value) = checked_value("probe_reads_log2", cfg.probe_reads_log2, 2..=5) {
            self.probe_reads_log2 = value;
        }
        if let Some(value) = checked_value(
            "probe_sweeps",
            cfg.probe_sweeps,
            1..=i64::try_from(MAX_PROBE_SWEEPS).expect("probe cap fits i64"),
        ) {
            self.probe_sweeps = usize::try_from(value).expect("probe cap fits usize");
        }
    }

    pub(crate) fn params(self) -> ScreenParams {
        ScreenParams {
            reads_log2: self.probe_reads_log2,
            sweeps: self.probe_sweeps,
            ..ScreenParams::default()
        }
    }
}

fn checked_value(
    name: &str,
    value: Option<i64>,
    range: std::ops::RangeInclusive<i64>,
) -> Option<u32> {
    let value = value?;
    if range.contains(&value) {
        Some(u32::try_from(value).expect("screen ranges fit u32"))
    } else {
        tracing::error!(value, "invalid screen.{name}; retaining previous value");
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    Keep,
    Audit,
    Reject,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct SelectorKey {
    topology_hash: Vec<u8>,
    num_nodes: usize,
    edges: std::sync::Arc<[(usize, usize)]>,
    allowed_h: Vec<i32>,
    allowed_j: Vec<i32>,
    reads_log2: u32,
    sweeps: usize,
}

impl SelectorKey {
    pub(crate) fn new(
        hash: &[u8],
        topology: &quip_solver_core::TopologyView,
        params: ScreenParams,
    ) -> Self {
        Self {
            topology_hash: hash.to_vec(),
            num_nodes: topology.num_nodes,
            edges: topology.edges.clone().into(),
            allowed_h: topology.allowed_h_milli.clone(),
            allowed_j: topology.allowed_j_milli.clone(),
            reads_log2: params.reads_log2,
            sweeps: params.sweeps,
        }
    }
}

#[derive(Debug)]
struct Selection {
    cutoff: Cutoff,
    kept: VecDeque<i64>,
    audits: u64,
    below_kept_median: u64,
}

impl Selection {
    fn new(config: ScreenConfig) -> Self {
        let d = f64::from(config.keep_denominator);
        Self {
            cutoff: Cutoff::new(CutoffConfig {
                keep: d,
                keep_min: (d / 4.0).max(2.0),
                keep_max: d * 4.0,
                k0: 1.0,
                ..CutoffConfig::default()
            }),
            kept: VecDeque::with_capacity(256),
            audits: 0,
            below_kept_median: 0,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct AuditObservation {
    pub(crate) energy_milli: i64,
    pub(crate) kept_median: Option<f64>,
    pub(crate) audits: u64,
    pub(crate) below_kept_median: u64,
}

/// Owned by the sampler behind one mutex, so leases share their estimators.
#[derive(Debug)]
pub(crate) struct Selector {
    config: ScreenConfig,
    selections: HashMap<SelectorKey, Selection>,
    audit_rng: u64,
}

impl Default for Selector {
    fn default() -> Self {
        // The CUDA backend runs on Linux. Read the OS once per sampler,
        // keeping the deterministic xorshift stream local thereafter.
        let mut bytes = [0; 8];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut bytes))
            .expect("OS entropy is available for screen audits");
        Self::with_seed(ScreenConfig::default(), u64::from_ne_bytes(bytes))
    }
}

impl Selector {
    pub(crate) fn with_seed(config: ScreenConfig, seed: u64) -> Self {
        Self {
            config,
            selections: HashMap::new(),
            audit_rng: seed.max(1),
        }
    }

    pub(crate) fn config(&self) -> ScreenConfig {
        self.config
    }

    pub(crate) fn reconfigure(&mut self, update: &ScreenToml) {
        let previous = self.config;
        self.config.merge(update);
        if previous.keep_denominator != self.config.keep_denominator
            || previous.probe_reads_log2 != self.config.probe_reads_log2
            || previous.probe_sweeps != self.config.probe_sweeps
        {
            self.selections.clear();
        }
    }

    /// Target hits always keep. Otherwise decide before observing, as Metal
    /// does. No GPU-share load updates or automatic audit adjustments run.
    #[allow(clippy::cast_precision_loss)] // Cutoff estimates use f64 milli-energies, as in Metal.
    pub(crate) fn decide(
        &mut self,
        key: &SelectorKey,
        energy_milli: i64,
        live_target: Option<i64>,
    ) -> Decision {
        if !self.config.enabled {
            return Decision::Keep;
        }
        let selection = self
            .selections
            .entry(key.clone())
            .or_insert_with(|| Selection::new(self.config));
        if live_target.is_some_and(|target| energy_milli <= target) {
            selection.cutoff.observe(energy_milli as f64);
            return Decision::Keep;
        }
        if selection.cutoff.decide(energy_milli as f64) {
            return Decision::Keep;
        }
        if self.config.audit_denominator != 0 {
            self.audit_rng ^= self.audit_rng << 13;
            self.audit_rng ^= self.audit_rng >> 7;
            self.audit_rng ^= self.audit_rng << 17;
            if self
                .audit_rng
                .is_multiple_of(u64::from(self.config.audit_denominator))
            {
                return Decision::Audit;
            }
        }
        Decision::Reject
    }

    /// Record completed full solves only. Missing entries belong to a reset
    /// configuration and cannot train the replacement estimator.
    #[allow(clippy::cast_precision_loss)] // The observational median follows Metal's f64 comparison.
    pub(crate) fn record_full(
        &mut self,
        key: &SelectorKey,
        decision: Decision,
        energy_milli: i64,
    ) -> Option<AuditObservation> {
        let selection = self.selections.get_mut(key)?;
        match decision {
            Decision::Keep => {
                if selection.kept.len() == 256 {
                    selection.kept.pop_front();
                }
                selection.kept.push_back(energy_milli);
                None
            }
            Decision::Audit => {
                let mut sorted: Vec<_> = selection.kept.iter().copied().collect();
                sorted.sort_unstable();
                let mid = sorted.len() / 2;
                let median = if sorted.is_empty() {
                    None
                } else if sorted.len().is_multiple_of(2) {
                    Some(sorted[mid - 1] as f64 / 2.0 + sorted[mid] as f64 / 2.0)
                } else {
                    Some(sorted[mid] as f64)
                };
                selection.audits += 1;
                selection.below_kept_median +=
                    u64::from(median.is_some_and(|m| (energy_milli as f64) < m));
                let observation = AuditObservation {
                    energy_milli,
                    kept_median: median,
                    audits: selection.audits,
                    below_kept_median: selection.below_kept_median,
                };
                tracing::debug!(?observation, "screen full-solve audit");
                Some(observation)
            }
            Decision::Reject => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> SelectorKey {
        SelectorKey::new(
            &[7; 32],
            &quip_solver_core::TopologyView {
                num_nodes: 4,
                edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
                allowed_h_milli: vec![0],
                allowed_j_milli: vec![-1000, 1000],
            },
            crate::screen::ScreenParams::default(),
        )
    }

    fn no_audits() -> ScreenConfig {
        ScreenConfig {
            audit_denominator: 0,
            ..ScreenConfig::default()
        }
    }

    #[test]
    fn target_hits_keep_before_warm_up_including_equality() {
        let mut selector = Selector::with_seed(no_audits(), 7);
        assert_eq!(selector.decide(&key(), -101, Some(-100)), Decision::Keep);
        assert_eq!(selector.decide(&key(), -100, Some(-100)), Decision::Keep);
        assert_eq!(selector.decide(&key(), -99, Some(-100)), Decision::Reject);
        // Target comparison stays exact even above f64's integer precision.
        assert_eq!(
            selector.decide(&key(), i64::MAX, Some(i64::MAX - 1)),
            Decision::Reject
        );
        assert_eq!(
            selector.decide(&key(), i64::MIN, Some(i64::MIN)),
            Decision::Keep
        );
    }

    #[test]
    fn target_hits_still_train_the_probe_distribution() {
        let mut selector = Selector::with_seed(no_audits(), 7);
        for _ in 0..200 {
            assert_eq!(selector.decide(&key(), 0, Some(0)), Decision::Keep);
        }
        assert_eq!(selector.decide(&key(), -1, None), Decision::Keep);
    }

    fn warm(selector: &mut Selector, key: &SelectorKey) {
        for _ in 0..200 {
            selector.decide(key, 0, None);
        }
        assert_eq!(selector.decide(key, -100, None), Decision::Keep);
    }

    #[test]
    fn selector_separates_every_key_component_and_reuses_equal_keys() {
        let mut selector = Selector::with_seed(no_audits(), 7);
        let original = key();
        warm(&mut selector, &original);
        let mut variants = vec![original.clone(); 7];
        variants[0].topology_hash[0] ^= 1;
        variants[1].allowed_h.push(1000);
        variants[2].allowed_j.reverse();
        variants[3].reads_log2 = 3;
        variants[4].sweeps = 1024;
        variants[5].num_nodes += 1;
        std::sync::Arc::make_mut(&mut variants[6].edges).swap(0, 1);
        for variant in variants {
            assert_eq!(selector.decide(&variant, -100, None), Decision::Reject);
        }
        assert_eq!(selector.decide(&key(), -100, None), Decision::Keep);
    }

    #[test]
    fn cutoff_config_changes_reset_existing_estimators() {
        for update in [
            "keep_denominator = 128",
            "probe_reads_log2 = 3",
            "probe_sweeps = 1024",
        ] {
            let mut selector = Selector::with_seed(no_audits(), 7);
            warm(&mut selector, &key());
            selector.reconfigure(&toml::from_str(update).unwrap());
            assert_eq!(
                selector.decide(&key(), -100, None),
                Decision::Reject,
                "{update}"
            );
        }
    }

    #[test]
    fn audit_and_enable_updates_preserve_estimators_and_disabled_keeps() {
        let mut selector = Selector::with_seed(no_audits(), 7);
        warm(&mut selector, &key());
        selector.reconfigure(
            &toml::from_str("audit_denominator = 200\nkeep_denominator = 16").unwrap(),
        );
        assert_eq!(selector.decide(&key(), -100, None), Decision::Keep);
        selector.reconfigure(&toml::from_str("enabled = false").unwrap());
        assert_eq!(selector.decide(&key(), 1_000_000, None), Decision::Keep);
        selector.reconfigure(&toml::from_str("enabled = true\naudit_denominator = 0").unwrap());
        assert_eq!(selector.decide(&key(), -100, None), Decision::Keep);
        assert_eq!(selector.decide(&key(), 1_000_000, None), Decision::Reject);
    }

    #[test]
    #[allow(clippy::float_cmp)] // A fixed zero load integral must remain exactly zero.
    fn full_solve_audits_compare_with_kept_median_without_feedback() {
        let mut selector = Selector::with_seed(no_audits(), 7);
        warm(&mut selector, &key());
        let before = selector.selections[&key()].cutoff.cutoff();
        assert_eq!(
            selector.record_full(&key(), Decision::Audit, -200),
            Some(AuditObservation {
                energy_milli: -200,
                kept_median: None,
                audits: 1,
                below_kept_median: 0,
            })
        );
        for energy in [-300, -100, -200, -400] {
            assert!(selector
                .record_full(&key(), Decision::Keep, energy)
                .is_none());
        }
        assert_eq!(
            selector.record_full(&key(), Decision::Audit, -300),
            Some(AuditObservation {
                energy_milli: -300,
                kept_median: Some(-250.0),
                audits: 2,
                below_kept_median: 1,
            })
        );
        assert!(selector
            .record_full(&key(), Decision::Reject, -1000)
            .is_none());
        let worse = selector.record_full(&key(), Decision::Audit, -100).unwrap();
        assert_eq!((worse.audits, worse.below_kept_median), (3, 1));
        assert_eq!(selector.selections[&key()].cutoff.cutoff(), before);
        assert_eq!(selector.selections[&key()].cutoff.log_adjust(), 0.0);
    }

    #[test]
    fn kept_full_energy_history_is_bounded_and_reset_drops_late_results() {
        let mut selector = Selector::with_seed(no_audits(), 7);
        warm(&mut selector, &key());
        for _ in 0..256 {
            selector.record_full(&key(), Decision::Keep, -1000);
        }
        for _ in 0..256 {
            selector.record_full(&key(), Decision::Keep, -100);
        }
        assert_eq!(
            selector
                .record_full(&key(), Decision::Audit, -101)
                .unwrap()
                .kept_median,
            Some(-100.0)
        );
        selector.reconfigure(&toml::from_str("keep_denominator = 128").unwrap());
        assert!(selector
            .record_full(&key(), Decision::Audit, -100)
            .is_none());
    }

    #[test]
    fn warm_up_rejects_and_decides_before_observing() {
        let mut selector = Selector::with_seed(no_audits(), 7);
        for _ in 0..199 {
            assert_eq!(selector.decide(&key(), 0, None), Decision::Reject);
        }
        assert_eq!(selector.decide(&key(), -1_000_000, None), Decision::Reject);
        assert_eq!(selector.decide(&key(), -1_000_000, None), Decision::Keep);
    }

    #[test]
    fn audit_stream_is_reproducible_and_matches_one_in_two_hundred() {
        let mut selector = Selector::with_seed(ScreenConfig::default(), 19);
        let mut same = Selector::with_seed(ScreenConfig::default(), 19);
        let mut audits = 0;
        for i in 0..100_000 {
            // Increasing energies always miss the lower-tail cutoff.
            let decision = selector.decide(&key(), i, None);
            assert_eq!(decision, same.decide(&key(), i, None));
            assert_ne!(decision, Decision::Keep);
            audits += usize::from(decision == Decision::Audit);
        }
        assert!((400..=600).contains(&audits), "{audits}");
    }

    #[test]
    fn audit_zero_disables_audits_even_for_a_zero_seed() {
        let mut selector = Selector::with_seed(no_audits(), 0);
        for i in 0..1000 {
            assert_eq!(selector.decide(&key(), i, None), Decision::Reject);
        }
        let mut selector = Selector::with_seed(ScreenConfig::default(), 0);
        let audits = (0..1000)
            .filter(|&i| selector.decide(&key(), i, None) == Decision::Audit)
            .count();
        assert!((1..100).contains(&audits), "{audits}");
    }

    #[test]
    // Box-Muller uses 53 random bits and rounds bounded test energies to milli.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn gaussian_keep_fraction_converges_near_configured_denominator() {
        let mut selector = Selector::with_seed(no_audits(), 7);
        let key = key();
        let mut rng = 9u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut kept = 0;
        for i in 0..200_000 {
            let (u, v) = (next(), next());
            let energy =
                -40_000.0 + 500.0 * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            let decision = selector.decide(&key, (energy * 1000.0).round() as i64, None);
            kept += usize::from(i >= 100_000 && decision == Decision::Keep);
        }
        let rate = f64::from(u32::try_from(kept).unwrap()) / 100_000.0;
        assert!((rate - 1.0 / 16.0).abs() < 0.1 / 16.0, "{rate}");
    }

    #[test]
    fn config_applies_valid_fields_and_keeps_omitted_values() {
        let mut config = ScreenConfig::default();
        let update: ScreenToml = toml::from_str(
            "enabled = false\nkeep_denominator = 128\naudit_denominator = 0\nprobe_reads_log2 = 5\nprobe_sweeps = 1",
        ).unwrap();
        config.merge(&update);
        assert_eq!(
            config,
            ScreenConfig {
                enabled: false,
                keep_denominator: 128,
                audit_denominator: 0,
                probe_reads_log2: 5,
                probe_sweeps: 1,
            }
        );
        config.merge(&ScreenToml::default());
        assert_eq!(config.keep_denominator, 128);
    }

    #[test]
    fn config_rejects_each_out_of_range_value_independently() {
        let mut config = ScreenConfig::default();
        for (key, values) in [
            ("keep_denominator", vec![-1, 0, 1, 100_001, i64::MAX]),
            ("audit_denominator", vec![-1, 1, 1_000_001, i64::MAX]),
            ("probe_reads_log2", vec![-1, 0, 1, 6, i64::MAX]),
            (
                "probe_sweeps",
                vec![
                    -1,
                    0,
                    i64::try_from(MAX_PROBE_SWEEPS).unwrap() + 1,
                    i64::MAX,
                ],
            ),
        ] {
            for value in values {
                config.merge(&toml::from_str(&format!("{key} = {value}")).unwrap());
                assert_eq!(config, ScreenConfig::default(), "{key} = {value}");
            }
        }
        config.merge(&toml::from_str("keep_denominator = 1\naudit_denominator = 2").unwrap());
        assert_eq!(config.keep_denominator, 16);
        assert_eq!(config.audit_denominator, 2);
    }

    #[test]
    fn config_accepts_inclusive_range_boundaries() {
        let mut config = ScreenConfig::default();
        config.merge(&toml::from_str(
            "keep_denominator = 2\naudit_denominator = 2\nprobe_reads_log2 = 2\nprobe_sweeps = 1",
        ).unwrap());
        assert_eq!(
            (
                config.keep_denominator,
                config.audit_denominator,
                config.probe_reads_log2,
                config.probe_sweeps
            ),
            (2, 2, 2, 1)
        );
        config.merge(&toml::from_str(&format!(
            "keep_denominator = 100000\naudit_denominator = 1000000\nprobe_reads_log2 = 5\nprobe_sweeps = {MAX_PROBE_SWEEPS}",
        )).unwrap());
        assert_eq!(
            (
                config.keep_denominator,
                config.audit_denominator,
                config.probe_reads_log2,
                config.probe_sweeps
            ),
            (100_000, 1_000_000, 5, MAX_PROBE_SWEEPS)
        );
    }
}

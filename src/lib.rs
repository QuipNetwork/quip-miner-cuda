//! CUDA Ising samplers.
//!
//! Three binaries share this library:
//! - `quip-cuda-sa` — Metropolis simulated annealing on one GPU
//! - `quip-cuda-msa` — multi-spin coded simulated annealing, 64 reads per word
//! - `quip-cuda-gibbs` — single-site heat-bath Gibbs on one GPU
//!
//! Kernels are the v0.2 self-feeding persistent kernels (`GPU/cuda_sa.cu` /
//! `GPU/cuda_gibbs.cu`, copied verbatim into `kernels/`): a kernel-side
//! nonce economy with 3-slot rotating buffers keeps many models in flight
//! across one persistent launch (see [`streaming`]). Solution energies are
//! always scored with [`quip_protocol::scoring::energy_milli`] for
//! consensus — the kernel's own int8-quantized energy tracking only drives
//! its internal annealing accept/reject decisions. The coordinator session
//! loop lives in `quip-solver-core`.

pub mod accept_table;
pub mod bench;
pub mod capacity;
pub mod coefficients;
pub mod corpus;
pub mod cuda_device;
// Preserve Metal's estimator arithmetic: counts intentionally become floats,
// and its probability midpoint stays in [0, 1]. Keep its exact-value tests.
#[allow(clippy::cast_precision_loss, clippy::manual_midpoint)]
#[cfg_attr(test, allow(clippy::float_cmp))]
mod cutoff;
pub mod driver_budget;
mod jit_cache;
pub mod kernel;
mod lease;
mod lease_turn;
pub mod model_file;
pub mod nsight;
pub mod nvml_gov;
pub mod probe;
pub mod sampler;
pub mod schema;
pub mod screen;
mod screen_policy;
mod stream_pool;
pub mod streaming;
pub mod topology;

pub use kernel::KernelKind;
pub use quip_solver_core::{IsingGraph, SampleParams, SamplerResult};
pub use sampler::sample_ising;
pub use streaming::{sample_one_real, RealSample, SampleSet};

use cuda_device::CudaDevice;
use nvml_gov::UtilGovernor;
use quip_solver_core::config::{config_override, warn_unknown_fields};
use quip_solver_core::quip_proto::v1::Backend;
use quip_solver_core::{
    BackendIdentity, CancelToken, ModelLimits, SampleError, Sampler, StreamJob, StreamResult,
    EXACT_COEFFICIENTS_FEATURE, LINEAR_SCHEDULE_FEATURE,
};
use std::collections::BTreeMap;
use std::sync::{Mutex, PoisonError};

/// MSA sampler with backend-owned salt leases and ordinary CUDA job handling.
pub struct MsaSampler {
    inner: CudaSampler,
    turns: lease_turn::LeaseTurns,
    logged_modes: Mutex<std::collections::HashSet<screen_policy::SelectorKey>>,
}

impl MsaSampler {
    /// Bind an MSA device and governor. Lease turns share this device.
    #[must_use]
    pub fn new(device: CudaDevice, gov: UtilGovernor) -> Self {
        Self {
            inner: CudaSampler::new(device, gov, KernelKind::Msa),
            turns: lease_turn::LeaseTurns::default(),
            logged_modes: Mutex::default(),
        }
    }
}

impl Sampler for MsaSampler {
    fn generates_locally() -> bool {
        true
    }

    fn sample_lease(
        &self,
        lease: &quip_solver_core::Lease,
        topology: &quip_solver_core::TopologyView,
        params: &SampleParams,
        sink: &quip_solver_core::LeaseSink,
    ) -> Result<(), SampleError> {
        lease::sample(self, lease, topology, params, sink)
    }

    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        self.inner.sample(graph, params)
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        self.inner.sample_stream(jobs, out, cancel);
    }

    fn accepts_warm_start() -> bool {
        CudaSampler::accepts_warm_start()
    }

    fn sample_warm(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        warm: &quip_solver_core::WarmStart,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        self.inner.sample_warm(graph, params, warm)
    }

    fn sample_stream_warm(
        &self,
        jobs: tokio::sync::mpsc::Receiver<quip_solver_core::WarmStreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        self.inner.sample_stream_warm(jobs, out, cancel);
    }

    fn stream_width(&self) -> usize {
        self.inner.stream_width()
    }

    fn declared_stream_width() -> u32 {
        CudaSampler::declared_stream_width()
    }

    fn utilization(&self) -> f64 {
        self.inner.utilization()
    }

    fn should_throttle(&self) -> bool {
        self.inner.should_throttle()
    }

    fn max_reads(&self) -> u32 {
        self.inner.max_reads()
    }

    fn apply_config(&self, backend_toml: &str) {
        self.inner.apply_config(backend_toml);
    }
}

const DEFAULT_MAX_EDGES: u32 = 1_000_000;

/// Features every kernel advertises: a real `sample_stream` override, the NVML
/// governor, exact coefficient uploads, and a host-built ladder that honours
/// `SampleParams::schedule`.
const FEATURES: &[&str] = &[
    "streaming",
    "governor",
    EXACT_COEFFICIENTS_FEATURE,
    LINEAR_SCHEDULE_FEATURE,
];

/// CUDA backend config, parsed from the verbatim `config.toml` subsection in
/// `Configure.backend_toml`. Unrecognized keys land in `unknown`;
/// [`warn_unknown_fields`] filters session-level keys (e.g. `num_sweeps`) so
/// only genuine typos are reported.
#[derive(serde::Deserialize, Default)]
struct CudaConfig {
    /// GPU utilization ceiling 1–100 (governor throttle threshold when yielding).
    utilization: Option<u32>,
    /// Yield the GPU to siblings when util exceeds the ceiling.
    yielding: Option<bool>,
    #[serde(default)]
    screen: screen_policy::ScreenToml,
    #[serde(flatten)]
    unknown: BTreeMap<String, toml::Value>,
}

/// CUDA adapt envelope (from `GPU/cuda_miner.py`).
const CUDA_ADAPT: quip_solver_core::adapt::AdaptBounds = quip_solver_core::adapt::AdaptBounds {
    min_sweeps: 256,
    max_sweeps: 2048,
    min_reads: 64,
    max_reads: 256,
    reads_solution_min_factor: 0,
    reads_solution_max_factor: 0,
    reads_solution_floor_factor: 0,
};

/// Narrow a resolved capacity to the wire type.
///
/// Saturating is safe: every value reaching here has passed
/// [`capacity::resolve`], which bounds it by kernel index widths or the
/// device memory budget, below `u32::MAX`.
fn identity_max_nodes(max_nodes: usize) -> u32 {
    u32::try_from(max_nodes).unwrap_or(u32::MAX)
}

/// Backend identity for `quip-cuda-sa` at a resolved capacity. `max_nodes` is
/// the self-feeding SA kernel's `unpacked_state` size (thread-local array in
/// `kernels/sa.cu`) — a job over this would overrun kernel-local storage, not
/// just run slowly, so it must reject `TooLarge` rather than clamp.
#[must_use]
pub fn cuda_sa_identity(max_nodes: usize) -> BackendIdentity {
    BackendIdentity {
        backend: Backend::Cuda,
        algorithm: KernelKind::Sa.algorithm(),
        max_nodes: identity_max_nodes(max_nodes),
        max_edges: DEFAULT_MAX_EDGES,
        features: FEATURES,
        adapt: CUDA_ADAPT,
        // SA has no degree bound (`Some(0)` is unbounded).
        limits: ModelLimits {
            max_degree: Some(0),
            // The wide path holds a local field up to WIDE_MAX_FIELD milli. Advertise
            // one milli less because quip-solver-core 0.0.3's --solve conversion
            // (encoding::float_milli) does not round: 536870.91 units does not multiply
            // to a whole milli in f64, while 536870.909 does.
            max_local_field_milli: Some(
                u32::try_from(coefficients::WIDE_MAX_FIELD - 1).unwrap_or(u32::MAX),
            ),
            exact_resolution_milli: Some(1),
        },
    }
}

/// Adapt envelope for `quip-cuda-msa` at `max_reads` reads per nonce.
///
/// Reads are pinned: `min_reads` equals `max_reads`, so the controller in
/// `quip_solver_core::adapt` hands out exactly this count rather than scaling
/// it with difficulty. The kernel packs 64 replicas into each 64-bit word, so
/// the count is 64 per replica word the device's opt-in shared memory holds —
/// 128 on a 99 KB card at Zephyr scale, 64 on a 64 KB Turing card.
/// `tests/headless_api.rs` pins the numbers together.
///
/// Sweeps are far cheaper than in `sa.cu` (one bitwise update serves 64
/// replicas), so the envelope runs deeper: measured on an RTX 5090 Laptop
/// against 24 `advantage2-system1` problems (MR !26), the deep-solution rate
/// keeps improving through 29568 sweeps.
#[must_use]
pub const fn cuda_msa_adapt(max_reads: u32) -> quip_solver_core::adapt::AdaptBounds {
    quip_solver_core::adapt::AdaptBounds {
        min_sweeps: 7392,
        max_sweeps: 29568,
        min_reads: max_reads,
        max_reads,
        reads_solution_min_factor: 0,
        reads_solution_max_factor: 0,
        reads_solution_floor_factor: 0,
    }
}

/// Backend identity for `quip-cuda-msa` at a resolved capacity.
///
/// Node and edge limits come from the device's Binary footprint. Spins past
/// shared memory spill to global memory. Each axis is an independent bound,
/// and the session checks the full model footprint before allocating.
#[must_use]
pub fn cuda_msa_identity(max_nodes: usize, max_edges: usize) -> BackendIdentity {
    BackendIdentity {
        backend: Backend::Cuda,
        algorithm: KernelKind::Msa.algorithm(),
        max_nodes: identity_max_nodes(max_nodes),
        max_edges: u32::try_from(max_edges).unwrap_or(u32::MAX),
        features: FEATURES,
        adapt: cuda_msa_adapt(u32::try_from(capacity::MSA_MAX_READS).unwrap_or(u32::MAX)),
        // Unit models run the unit-weight kernel (by sign, 64 lanes per word).
        // Other models run the weighted kernel at the model's own scale, up to a
        // local field of 2^24 - 1 milli at scale 1, the worst case.
        limits: ModelLimits {
            max_degree: None,
            max_local_field_milli: Some(
                u32::try_from(coefficients::MSA_WEIGHTED_MAX_FIELD).unwrap_or(u32::MAX),
            ),
            exact_resolution_milli: Some(1),
        },
    }
}

/// Backend identity for `quip-cuda-gibbs` at a resolved capacity. `max_nodes`
/// is the self-feeding Gibbs kernel's `shared_state` size in
/// `kernels/gibbs.cu`.
#[must_use]
pub fn cuda_gibbs_identity(max_nodes: usize) -> BackendIdentity {
    BackendIdentity {
        backend: Backend::Cuda,
        algorithm: KernelKind::Gibbs.algorithm(),
        max_nodes: identity_max_nodes(max_nodes),
        max_edges: DEFAULT_MAX_EDGES,
        features: FEATURES,
        adapt: CUDA_ADAPT,
        // The int8 bound is per coefficient at the model's own scale, which no
        // binary-wide field limit expresses: the binary refuses per job.
        limits: ModelLimits {
            max_degree: Some(0),
            max_local_field_milli: None,
            exact_resolution_milli: Some(1),
        },
    }
}

/// CUDA sampler backend: one GPU device plus an NVML utilization governor.
#[derive(Debug)]
pub struct CudaSampler {
    device: CudaDevice,
    gov: UtilGovernor,
    kernel: KernelKind,
    screen: Mutex<screen_policy::Selector>,
}

impl CudaSampler {
    /// Bind an open device and its NVML governor into a sampler for `kernel`.
    #[must_use]
    pub fn new(device: CudaDevice, gov: UtilGovernor, kernel: KernelKind) -> Self {
        Self {
            device,
            gov,
            kernel,
            screen: Mutex::new(screen_policy::Selector::default()),
        }
    }
}

impl Sampler for CudaSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        // quip-miner-cuda-gp4: the device-error -> wire-error mapping lives on
        // `sampler::SampleError`'s `From` impl (TYPE-4: exhaustive, no wildcard).
        sample_ising(&self.device, graph, params, self.kernel).map_err(|e| {
            eprintln!("cuda sample failed: {e}");
            e.into()
        })
    }

    /// Self-feeding kernel instances: `max_sms / sms_per_nonce` (1 for SA and
    /// msa, 4 for Gibbs), each a fully independent nonce group.
    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        streaming::run_stream(&self.device, self.kernel, jobs, out, cancel, &self.gov);
    }

    fn stream_width(&self) -> usize {
        streaming::stream_width(&self.device, self.kernel)
    }

    /// The live width is `max_sms / sms_per_nonce` — a property of the opened
    /// device — and `--capabilities` must answer without opening one, so no
    /// honest device-independent number exists. Declare the 0 sentinel
    /// (quip-solver-core 0.0.1-rc1): width unknown until the device opens.
    /// The in-session `Capabilities` reply then carries the live
    /// [`Self::stream_width`], and the session logs no misdeclaration.
    fn declared_stream_width() -> u32 {
        0
    }

    fn utilization(&self) -> f64 {
        f64::from(self.gov.utilization())
    }

    fn should_throttle(&self) -> bool {
        self.gov.should_throttle()
    }

    fn max_reads(&self) -> u32 {
        if self.kernel == KernelKind::Msa {
            u32::try_from(capacity::msa_device_reads(&self.device.limits)).unwrap_or(128)
        } else {
            streaming::max_reads(self.kernel)
        }
    }

    fn apply_config(&self, backend_toml: &str) {
        // quip-miner-cuda-9yk: log parse errors before collapsing to defaults
        // so operator typos are visible next to warn_unknown_fields.
        let cfg: CudaConfig = match toml::from_str(backend_toml) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("cuda backend_toml parse failed, using defaults: {e}");
                CudaConfig::default()
            }
        };
        warn_unknown_fields("cuda", cfg.unknown.keys());
        warn_unknown_fields(
            "cuda",
            cfg.screen.unknown.keys().map(|key| format!("screen.{key}")),
        );
        self.screen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .reconfigure(&cfg.screen);
        // config over CLI (the governor holds the CLI-set values until now).
        let ceiling = config_override(
            "utilization",
            self.gov.utilization_ceiling(),
            cfg.utilization,
        );
        let yielding = config_override("yielding", self.gov.yielding(), cfg.yielding);
        self.gov.reconfigure(ceiling, yielding);
    }
}

#[cfg(test)]
mod config_tests {
    use super::CudaConfig;
    use crate::screen_policy::Selector;
    use proptest::prelude::*;

    #[test]
    fn parses_known_fields_and_collects_unknown() {
        let cfg: CudaConfig =
            toml::from_str("utilization = 60\nyielding = true\nnum_sweeps = 100\nfoo = 1\n")
                .unwrap();
        assert_eq!(cfg.utilization, Some(60));
        assert_eq!(cfg.yielding, Some(true));
        // num_sweeps + foo are not CUDA fields -> flattened into `unknown`;
        // warn_unknown_fields later drops num_sweeps (a session key), keeps foo.
        let unknown: Vec<&str> = cfg.unknown.keys().map(String::as_str).collect();
        assert_eq!(unknown, vec!["foo", "num_sweeps"]);
    }

    #[test]
    fn empty_config_is_all_none() {
        let cfg: CudaConfig = toml::from_str("").unwrap();
        assert!(cfg.utilization.is_none());
        assert!(cfg.yielding.is_none());
        assert!(cfg.unknown.is_empty());
    }

    #[test]
    fn nested_screen_config_updates_live_values_and_collects_unknown_keys() {
        let cfg: CudaConfig = toml::from_str(
            "utilization = 60\n[screen]\nenabled = false\nkeep_denominator = 128\naudit_denominator = 0\nprobe_reads_log2 = 3\nprobe_sweeps = 1024\nprobe_sweep = 99\nnum_sweeps = 7",
        ).unwrap();
        assert_eq!(cfg.utilization, Some(60));
        assert!(cfg.unknown.is_empty());
        assert_eq!(
            cfg.screen
                .unknown
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["num_sweeps", "probe_sweep"]
        );
        let mut selector = Selector::with_seed(crate::screen_policy::ScreenConfig::default(), 7);
        selector.reconfigure(&cfg.screen);
        let live = selector.config();
        assert!(!live.enabled);
        assert_eq!((live.keep_denominator, live.audit_denominator), (128, 0));
        assert_eq!(
            (
                live.params().reads_log2,
                live.params().sweeps,
                live.params().threads
            ),
            (3, 1024, 256)
        );
        let invalid: CudaConfig = toml::from_str("[screen]\nkeep_denominator = -1\naudit_denominator = 1000001\nprobe_reads_log2 = 6\nprobe_sweeps = 0").unwrap();
        selector.reconfigure(&invalid.screen);
        assert_eq!(selector.config(), live);
        selector.reconfigure(&toml::from_str::<CudaConfig>("").unwrap().screen);
        assert_eq!(selector.config(), live);
    }

    /// Parse path used by `apply_config` must never panic on arbitrary input.
    #[test]
    fn parse_never_panics_on_arbitrary_string() {
        proptest!(|(s in ".*")| {
            let _cfg: CudaConfig = toml::from_str(&s).unwrap_or_default();
        });
    }

    /// Same guarantee for arbitrary TOML tables (structured, not just strings).
    #[test]
    fn parse_never_panics_on_arbitrary_toml_table() {
        proptest!(|(entries in prop::collection::btree_map(
            "[a-zA-Z_][a-zA-Z0-9_]{0,12}",
            prop_oneof![
                Just(toml::Value::Boolean(true)),
                Just(toml::Value::Boolean(false)),
                (0i64..10_000).prop_map(toml::Value::Integer),
                ".*".prop_map(toml::Value::String),
            ],
            0..12,
        ))| {
            // toml 0.8 stores tables as toml::map::Map, not BTreeMap.
            let mut map = toml::map::Map::new();
            for (k, v) in entries {
                map.insert(k, v);
            }
            let s = toml::to_string(&toml::Value::Table(map)).unwrap_or_default();
            let _cfg: CudaConfig = toml::from_str(&s).unwrap_or_default();
        });
    }

    /// Known fields round-trip through TOML parse.
    #[test]
    fn utilization_yielding_round_trip() {
        proptest!(|(
            util in prop::option::of(0u32..=100u32),
            yielding in prop::option::of(any::<bool>()),
        )| {
            let mut parts = Vec::new();
            if let Some(u) = util {
                parts.push(format!("utilization = {u}"));
            }
            if let Some(y) = yielding {
                parts.push(format!("yielding = {y}"));
            }
            let s = parts.join("\n");
            let cfg: CudaConfig = toml::from_str(&s).expect("valid constructed toml");
            prop_assert_eq!(cfg.utilization, util);
            prop_assert_eq!(cfg.yielding, yielding);
        });
    }

    /// Unrecognized keys always land in `unknown` and never vanish.
    #[test]
    fn unrecognized_keys_land_in_unknown() {
        proptest!(|(
            key in "[a-z][a-z0-9_]{0,15}",
            val in -1000i64..1000i64,
        )| {
            prop_assume!(key != "utilization" && key != "yielding" && key != "screen");
            let s = format!("{key} = {val}");
            let cfg: CudaConfig = toml::from_str(&s).expect("valid constructed toml");
            prop_assert!(
                cfg.unknown.contains_key(&key),
                "key {key:?} vanished from unknown"
            );
            prop_assert_eq!(cfg.unknown.get(&key), Some(&toml::Value::Integer(val)));
            // Known fields stay unset when only unknowns are present.
            prop_assert_eq!(cfg.utilization, None);
            prop_assert_eq!(cfg.yielding, None);
        });
    }

    /// Parse failure collapses to a full Default, never a partial config.
    #[test]
    fn parse_failure_collapses_to_default() {
        proptest!(|(s in ".*")| {
            if toml::from_str::<CudaConfig>(&s).is_err() {
                {
                    let cfg: CudaConfig = toml::from_str(&s).unwrap_or_default();
                    prop_assert_eq!(cfg.utilization, None);
                    prop_assert_eq!(cfg.yielding, None);
                    prop_assert!(cfg.unknown.is_empty());
                    // Explicitly equal to a fresh Default (no partial fields).
                    let def = CudaConfig::default();
                    prop_assert_eq!(cfg.utilization, def.utilization);
                    prop_assert_eq!(cfg.yielding, def.yielding);
                    prop_assert_eq!(cfg.unknown, def.unknown);
                }
            }
        });
    }

    /// Malformed known-field types must not partially apply siblings.
    #[test]
    fn type_mismatch_is_full_default_not_partial() {
        // utilization valid type would be int; string forces whole-document Err.
        let s = "utilization = \"nope\"\nyielding = true\n";
        assert!(toml::from_str::<CudaConfig>(s).is_err());
        let cfg: CudaConfig = toml::from_str(s).unwrap_or_default();
        assert!(cfg.utilization.is_none());
        assert!(cfg.yielding.is_none());
        assert!(cfg.unknown.is_empty());
    }
}

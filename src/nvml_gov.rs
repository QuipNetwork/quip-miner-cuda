//! NVML utilization governor (port of `GPU/gpu_scheduler.py` yielding path).
//!
//! Util ceiling + yielding are runtime knobs (atomics): the CLI sets them at
//! launch, and [`UtilGovernor::reconfigure`] overrides them when the
//! coordinator's `Configure` arrives. When yielding and *another process's*
//! GPU load exceeds the ceiling, the streaming loop ends its session so the
//! sibling user gets the SMs.
//!
//! # Two utilization figures, deliberately
//!
//! The governor keeps whole-device utilization and foreign utilization apart,
//! because they answer different questions and conflating them was QUI-882:
//!
//! * [`UtilGovernor::utilization`] — whole-device percent, what an operator
//!   expects to see in a `Status` message. Includes our own kernels.
//! * [`UtilGovernor::foreign_utilization`] — load attributable to *other*
//!   processes, and the only figure [`UtilGovernor::should_throttle`] reads.
//!
//! NVML's device-wide figure counts the caller's own kernels, so a working
//! miner holds it near 100% by construction. Throttling on it means the miner
//! detects its own load, concludes the GPU is contended, and yields to itself
//! forever — which is what QUI-882 measured as a 24x collapse on light
//! workloads. Foreign utilization is what "is someone else using this GPU?"
//! actually needs.
//!
//! See [`Attribution`] for how foreign load is established, and why the
//! process-count check is the part that holds in every environment.

use nvml_wrapper::Device;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// How a poll established the foreign load figure.
///
/// Recorded so an operator can tell a precise measurement from a conservative
/// one — on WSL2 and inside PID namespaces the precise path is unavailable,
/// and silently degrading without saying so is how QUI-882 stayed hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Attribution {
    /// Fewer than two compute processes hold a context, so there is nobody to
    /// yield to and foreign load is zero by construction. This check needs no
    /// PID matching, so it is the one that holds everywhere.
    Alone,
    /// `nvmlDeviceGetProcessUtilization` attributed load per PID and our own
    /// PID was among the device's processes, so subtracting ourselves is
    /// meaningful. The precise path.
    PerProcess,
    /// Another process is present but per-PID attribution is unusable — a PID
    /// namespace (the miner ships in Docker), MPS, or a driver that does not
    /// implement the call. Falls back to whole-device utilization, which
    /// over-counts by including our own load, but only ever while genuinely
    /// sharing, so it cannot fire when we are alone.
    WholeDevice,
}

/// Reconfigurable governor knobs plus the latest util samples, shared with the
/// poll thread.
struct Knobs {
    /// Util ceiling 1–100; throttle fires above it when yielding.
    ceiling: AtomicU32,
    yielding: AtomicBool,
    /// Last whole-device NVML GPU util percent 0–100, including our own load.
    /// Sampled whatever the yielding setting, because `Status` reports it.
    last_util: AtomicU32,
    /// Last foreign (other-process) GPU util percent 0–100. Zero unless
    /// yielding, since nothing reads it otherwise.
    last_foreign_util: AtomicU32,
    stop: AtomicBool,
}

/// Shared utilization sample and reconfigurable governor knobs.
pub struct UtilGovernor {
    knobs: Arc<Knobs>,
    handle: Option<JoinHandle<()>>,
}

// Governor knobs only; the NVML poll thread's `JoinHandle` is deliberately
// omitted (opaque handle state, no diagnostic value).
impl fmt::Debug for UtilGovernor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UtilGovernor")
            .field("ceiling", &self.utilization_ceiling())
            .field("yielding", &self.yielding())
            .field("last_util", &self.knobs.last_util.load(Ordering::Relaxed))
            .field(
                "last_foreign_util",
                &self.knobs.last_foreign_util.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl UtilGovernor {
    /// Start the NVML poller. Values come from the CLI; `Configure` may later
    /// override them via [`reconfigure`](Self::reconfigure). The poll thread
    /// runs regardless of `yielding` (so a later `false -> true` override starts
    /// sampling with no thread churn) but only records util while yielding.
    ///
    /// Falls back to a silent no-op if NVML init fails (miner still runs; util
    /// stays 0 and throttle never fires).
    ///
    /// The ceiling is clamped into `1..=100` on the way in:
    ///
    /// ```
    /// use quip_miner_cuda::nvml_gov::UtilGovernor;
    ///
    /// let gov = UtilGovernor::start(0, 90, true);
    /// assert_eq!(gov.utilization_ceiling(), 90);
    /// assert!(gov.yielding());
    ///
    /// // 0 clamps up to 1, 200 clamps down to 100.
    /// assert_eq!(UtilGovernor::start(0, 0, false).utilization_ceiling(), 1);
    /// assert_eq!(UtilGovernor::start(0, 200, false).utilization_ceiling(), 100);
    /// ```
    #[must_use]
    pub fn start(device_index: u32, utilization_ceiling: u32, yielding: bool) -> Self {
        let knobs = Arc::new(Knobs {
            ceiling: AtomicU32::new(utilization_ceiling.clamp(1, 100)),
            yielding: AtomicBool::new(yielding),
            last_util: AtomicU32::new(0),
            last_foreign_util: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        });
        let knobs_thread = Arc::clone(&knobs);
        let handle = Some(thread::spawn(move || {
            poll_loop(device_index, &knobs_thread);
        }));
        Self { knobs, handle }
    }

    /// Override the ceiling and yielding flag at runtime (config over CLI).
    ///
    /// The ceiling is clamped into `1..=100`, exactly as in
    /// [`start`](Self::start):
    ///
    /// ```
    /// use quip_miner_cuda::nvml_gov::UtilGovernor;
    ///
    /// let gov = UtilGovernor::start(0, 90, true);
    ///
    /// gov.reconfigure(0, false);
    /// assert_eq!(gov.utilization_ceiling(), 1);
    /// assert!(!gov.yielding());
    ///
    /// gov.reconfigure(200, true);
    /// assert_eq!(gov.utilization_ceiling(), 100);
    /// assert!(gov.yielding());
    /// ```
    pub fn reconfigure(&self, utilization_ceiling: u32, yielding: bool) {
        self.knobs
            .ceiling
            .store(utilization_ceiling.clamp(1, 100), Ordering::Relaxed);
        self.knobs.yielding.store(yielding, Ordering::Relaxed);
    }

    /// Current ceiling (CLI value, or the config override once applied).
    ///
    /// ```
    /// use quip_miner_cuda::nvml_gov::UtilGovernor;
    ///
    /// let gov = UtilGovernor::start(0, 75, false);
    /// assert_eq!(gov.utilization_ceiling(), 75);
    ///
    /// gov.reconfigure(40, false);
    /// assert_eq!(gov.utilization_ceiling(), 40);
    /// ```
    #[must_use]
    pub fn utilization_ceiling(&self) -> u32 {
        self.knobs.ceiling.load(Ordering::Relaxed)
    }

    /// Current yielding flag (CLI value, or the config override once applied).
    ///
    /// ```
    /// use quip_miner_cuda::nvml_gov::UtilGovernor;
    ///
    /// let gov = UtilGovernor::start(0, 75, false);
    /// assert!(!gov.yielding());
    ///
    /// gov.reconfigure(75, true);
    /// assert!(gov.yielding());
    /// ```
    #[must_use]
    pub fn yielding(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
    }

    /// Whole-device NVML GPU util percent (0–100), including our own kernels.
    ///
    /// This is the figure `Status` reports, so it is sampled whatever the
    /// yielding setting — an operator asking "is my GPU busy?" wants the real
    /// device number, and reporting 0 unless `--yielding` happened to be on
    /// left the default configuration with no utilization telemetry at all.
    ///
    /// Do not throttle on this. It counts our own load, so a working miner
    /// pins it near 100%; see [`Self::foreign_utilization`].
    #[must_use]
    pub fn utilization(&self) -> f32 {
        Self::as_percent(self.knobs.last_util.load(Ordering::Relaxed))
    }

    /// GPU util percent (0–100) attributable to processes other than this one.
    ///
    /// Zero when nothing else holds a context on the device, so a miner alone
    /// on its GPU reads zero here no matter how hard it is working. Zero while
    /// not yielding, since nothing consumes it then.
    #[must_use]
    pub fn foreign_utilization(&self) -> f32 {
        Self::as_percent(self.knobs.last_foreign_util.load(Ordering::Relaxed))
    }

    /// Clamp a raw NVML percent into `0..=100` and widen it losslessly.
    fn as_percent(raw: u32) -> f32 {
        let util = raw.min(100);
        f32::from(u8::try_from(util).unwrap_or(0))
    }

    /// True when yielding and *another process's* load exceeds the ceiling.
    ///
    /// Reads [`Self::foreign_utilization`], never the whole-device figure:
    /// throttling on device-wide util means throttling against our own kernels
    /// (QUI-882).
    ///
    /// With yielding off it is always false, whatever the last sample was:
    ///
    /// ```
    /// use quip_miner_cuda::nvml_gov::UtilGovernor;
    ///
    /// let gov = UtilGovernor::start(0, 50, false);
    /// assert!(!gov.should_throttle());
    /// ```
    #[must_use]
    pub fn should_throttle(&self) -> bool {
        throttle_decision(
            self.knobs.yielding.load(Ordering::Relaxed),
            self.knobs.last_foreign_util.load(Ordering::Relaxed),
            self.knobs.ceiling.load(Ordering::Relaxed),
        )
    }

    /// Request the poller to exit and join it.
    pub fn stop(&mut self) {
        self.knobs.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            // The payload is dropped rather than propagated: a dead poll
            // thread only costs util sampling, which the governor already
            // degrades gracefully to (util reads 0, throttle never fires), and
            // `stop` also runs from `drop`, where a panic would abort. Report
            // it so the loss of sampling is not silent.
            if h.join().is_err() {
                eprintln!("cuda util governor: NVML poll thread panicked; sampling stopped");
            }
        }
    }
}

impl Drop for UtilGovernor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Sum the SM utilization of every process except `own_pid`, and advance
/// `last_seen` past every sample considered.
///
/// NVML hands back *every* sample buffered since `last_seen`, which for a
/// long-running neighbour is several. Summing them raw would count that one
/// process once per buffered sample and manufacture contention that is not
/// there, so only each PID's newest sample counts.
fn foreign_sm_util(
    samples: &[nvml_wrapper::struct_wrappers::device::ProcessUtilizationSample],
    own_pid: u32,
    last_seen: &mut u64,
) -> u32 {
    let mut newest: HashMap<u32, (u64, u32)> = HashMap::new();
    for s in samples {
        *last_seen = (*last_seen).max(s.timestamp);
        newest
            .entry(s.pid)
            .and_modify(|cur| {
                if s.timestamp > cur.0 {
                    *cur = (s.timestamp, s.sm_util);
                }
            })
            .or_insert((s.timestamp, s.sm_util));
    }
    let foreign: u32 = newest
        .iter()
        .filter(|(pid, _)| **pid != own_pid)
        .map(|(_, (_, sm))| *sm)
        .sum();
    // Several busy neighbours can sum past 100; the ceiling is a percent, so
    // clamp rather than let the comparison see an impossible figure.
    foreign.min(100)
}

/// Whether the driver should give up its session, given the current knobs.
///
/// Split out from [`UtilGovernor::should_throttle`] so the decision is
/// testable without a GPU, a second process, or a race against the poll
/// thread overwriting the sample mid-assertion.
const fn throttle_decision(yielding: bool, foreign_util: u32, ceiling: u32) -> bool {
    yielding && foreign_util > ceiling
}

/// Load attributable to processes other than `own_pid`, and how it was found.
///
/// `last_seen` carries the newest NVML sample timestamp between calls so each
/// poll only reads samples produced since the previous one.
fn foreign_utilization(
    device: &Device<'_>,
    own_pid: u32,
    last_seen: &mut u64,
) -> (u32, Attribution) {
    // A miner that is streaming always holds a CUDA context, so fewer than two
    // compute processes means the only load on this device is ours. This is
    // the check that fixes QUI-882, and the only one that needs no PID
    // matching — so it survives PID namespaces, where everything below does
    // not.
    let processes = device.running_compute_processes().unwrap_or_default();
    if processes.len() < 2 {
        return (0, Attribution::Alone);
    }

    // Per-PID subtraction is only meaningful if we can find ourselves in the
    // list. Inside a container NVML reports host PIDs that never equal
    // `std::process::id()`, and then "no sample is ours" would read as "all of
    // this load is foreign" — the self-reflection bug wearing a new hat.
    if processes.iter().any(|p| p.pid == own_pid) {
        if let Ok(samples) = device.process_utilization_stats(*last_seen) {
            return (
                foreign_sm_util(&samples, own_pid, last_seen),
                Attribution::PerProcess,
            );
        }
    }

    // Someone else is here but we cannot size their share. Whole-device util
    // over-counts (it includes us), but it can only be reached while genuinely
    // sharing, so it never fires for a miner alone on its GPU.
    let util = device.utilization_rates().map_or(0, |r| r.gpu);
    (util, Attribution::WholeDevice)
}

fn poll_loop(device_index: u32, knobs: &Knobs) {
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        return;
    };
    let Ok(device) = nvml.device_by_index(device_index) else {
        return;
    };
    let own_pid = std::process::id();
    let mut last_seen = 0u64;
    let mut reported: Option<Attribution> = None;

    while !knobs.stop.load(Ordering::Relaxed) {
        // Sampled unconditionally: `Status` reports this whether or not the
        // operator asked for yielding.
        if let Ok(rates) = device.utilization_rates() {
            knobs.last_util.store(rates.gpu, Ordering::Relaxed);
        }

        if knobs.yielding.load(Ordering::Relaxed) {
            let (foreign, how) = foreign_utilization(&device, own_pid, &mut last_seen);
            knobs.last_foreign_util.store(foreign, Ordering::Relaxed);
            // Announce the attribution path once, and again whenever it
            // changes. An operator on WSL2 or in a container needs to know the
            // governor fell back to the over-counting figure, and a silent
            // degrade is how QUI-882 went unnoticed for so long.
            if reported != Some(how) {
                tracing::info!(
                    attribution = ?how,
                    foreign_util = foreign,
                    "cuda util governor: foreign-load attribution"
                );
                reported = Some(how);
            }
        } else {
            knobs.last_foreign_util.store(0, Ordering::Relaxed);
        }
        thread::sleep(Duration::from_secs(2));
    }
}

#[cfg(test)]
mod governor_tests {
    use super::{foreign_sm_util, throttle_decision, UtilGovernor};
    use nvml_wrapper::struct_wrappers::device::ProcessUtilizationSample;
    use std::sync::atomic::Ordering;

    fn sample(pid: u32, timestamp: u64, sm_util: u32) -> ProcessUtilizationSample {
        ProcessUtilizationSample {
            pid,
            timestamp,
            sm_util,
            mem_util: 0,
            enc_util: 0,
            dec_util: 0,
        }
    }

    /// The QUI-882 regression, stated directly: a miner saturating its own GPU
    /// must not throttle. Before the fix the decision read whole-device
    /// utilization, which a working miner pins near 100%, so this configuration
    /// throttled permanently and cost light workloads ~95% of their throughput.
    #[test]
    fn a_saturated_gpu_with_no_foreign_load_does_not_throttle() {
        // Whole device at 100%, but all of it ours.
        assert!(!throttle_decision(true, 0, 90));
        // Even with the ceiling as low as it goes.
        assert!(!throttle_decision(true, 0, 1));
    }

    #[test]
    fn foreign_load_above_the_ceiling_throttles_and_below_it_does_not() {
        assert!(throttle_decision(true, 91, 90));
        assert!(!throttle_decision(true, 90, 90), "ceiling is exclusive");
        assert!(!throttle_decision(true, 40, 90));
        // Yielding off short-circuits regardless of foreign load.
        assert!(!throttle_decision(false, 100, 1));
    }

    #[test]
    fn our_own_samples_never_count_as_foreign_load() {
        let mut last_seen = 0;
        let samples = [sample(42, 10, 80), sample(7, 10, 15)];

        assert_eq!(foreign_sm_util(&samples, 42, &mut last_seen), 15);
        // Seen from the neighbour's side, the same data attributes the other way.
        let mut other = 0;
        assert_eq!(foreign_sm_util(&samples, 7, &mut other), 80);
    }

    #[test]
    fn only_each_pids_newest_sample_counts() {
        // NVML buffers every sample since `last_seen`; a neighbour steady at
        // 30% must read as 30, not as the sum of its buffered samples.
        let mut last_seen = 0;
        let samples = [
            sample(7, 10, 30),
            sample(7, 20, 30),
            sample(7, 30, 30),
            sample(42, 30, 60),
        ];

        assert_eq!(foreign_sm_util(&samples, 42, &mut last_seen), 30);
        assert_eq!(last_seen, 30, "last_seen advances past every sample read");
    }

    #[test]
    fn several_busy_neighbours_clamp_to_a_percent() {
        let mut last_seen = 0;
        let samples = [sample(1, 5, 70), sample(2, 5, 70), sample(3, 5, 70)];

        assert_eq!(foreign_sm_util(&samples, 42, &mut last_seen), 100);
    }

    #[test]
    fn no_samples_means_no_foreign_load() {
        let mut last_seen = 99;
        assert_eq!(foreign_sm_util(&[], 42, &mut last_seen), 0);
        assert_eq!(last_seen, 99, "an empty read must not rewind last_seen");
    }

    // `poll_loop` returns immediately when NVML is absent, so the governor is
    // fully constructible headless and every knob below is testable without a
    // GPU.

    #[test]
    fn start_clamps_the_ceiling_into_1_100() {
        let below = UtilGovernor::start(0, 0, false);
        let above = UtilGovernor::start(0, 200, false);
        let inside = UtilGovernor::start(0, 55, false);

        assert_eq!(below.utilization_ceiling(), 1);
        assert_eq!(above.utilization_ceiling(), 100);
        assert_eq!(inside.utilization_ceiling(), 55);
    }

    #[test]
    fn reconfigure_clamps_and_round_trips_both_knobs() {
        let gov = UtilGovernor::start(0, 90, true);

        gov.reconfigure(0, false);
        assert_eq!(gov.utilization_ceiling(), 1);
        assert!(!gov.yielding());

        gov.reconfigure(200, true);
        assert_eq!(gov.utilization_ceiling(), 100);
        assert!(gov.yielding());

        gov.reconfigure(64, false);
        assert_eq!(gov.utilization_ceiling(), 64);
        assert!(!gov.yielding());
    }

    #[test]
    fn never_throttles_while_not_yielding() {
        let gov = UtilGovernor::start(0, 1, false);
        // Plant a sample far above the ceiling. On a host with NVML the poll
        // thread may overwrite it with 0, but either way `should_throttle`
        // short-circuits on the yielding flag, so the assertion holds under
        // every interleaving.
        gov.knobs.last_util.store(100, Ordering::Relaxed);
        assert!(!gov.should_throttle());
    }
}

//! NVML utilization governor (port of `GPU/gpu_scheduler.py` yielding path).
//!
//! Util ceiling + yielding are runtime knobs (atomics): the CLI sets them at
//! launch, and [`UtilGovernor::reconfigure`] overrides them when the
//! coordinator's `Configure` arrives. When yielding and the observed GPU util
//! exceeds the ceiling, the session loop inserts a brief pause so sibling GPU
//! users get time slices.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Reconfigurable governor knobs plus the latest util sample, shared with the
/// poll thread.
struct Knobs {
    /// Util ceiling 1–100; throttle fires above it when yielding.
    ceiling: AtomicU32,
    yielding: AtomicBool,
    /// Last NVML GPU util percent 0–100 (0 while not yielding).
    last_util: AtomicU32,
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
    pub fn start(device_index: u32, utilization_ceiling: u32, yielding: bool) -> Self {
        let knobs = Arc::new(Knobs {
            ceiling: AtomicU32::new(utilization_ceiling.clamp(1, 100)),
            yielding: AtomicBool::new(yielding),
            last_util: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        });
        let knobs_thread = Arc::clone(&knobs);
        let handle = Some(thread::spawn(move || {
            poll_loop(device_index, &knobs_thread)
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
    pub fn yielding(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
    }

    /// Last NVML GPU util percent (0–100), or 0 if not yielding / unavailable.
    pub fn utilization(&self) -> f32 {
        self.knobs.last_util.load(Ordering::Relaxed) as f32
    }

    /// True when yielding and the last util sample exceeds the ceiling.
    ///
    /// With yielding off it is always false, whatever the last sample was:
    ///
    /// ```
    /// use quip_miner_cuda::nvml_gov::UtilGovernor;
    ///
    /// let gov = UtilGovernor::start(0, 50, false);
    /// assert!(!gov.should_throttle());
    /// ```
    pub fn should_throttle(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
            && self.knobs.last_util.load(Ordering::Relaxed)
                > self.knobs.ceiling.load(Ordering::Relaxed)
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

fn poll_loop(device_index: u32, knobs: &Knobs) {
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        return;
    };
    let Ok(device) = nvml.device_by_index(device_index) else {
        return;
    };
    while !knobs.stop.load(Ordering::Relaxed) {
        if knobs.yielding.load(Ordering::Relaxed) {
            if let Ok(rates) = device.utilization_rates() {
                knobs.last_util.store(rates.gpu, Ordering::Relaxed);
            }
        } else {
            knobs.last_util.store(0, Ordering::Relaxed);
        }
        thread::sleep(Duration::from_secs(2));
    }
}

#[cfg(test)]
mod governor_tests {
    use super::UtilGovernor;
    use std::sync::atomic::Ordering;

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

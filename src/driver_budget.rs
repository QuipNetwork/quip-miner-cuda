//! Opt-in per-window time accounting for the production stream driver.
//!
//! This answers QUI-870's question — *which bucket's share grows as `att/s`
//! falls?* — for the Rust driver loop. The original instrument
//! (`GPU/driver_budget.py`) measured a Python stream driver that v0.3 replaced,
//! so the buckets are re-derived here against [`crate::streaming::pump_session`]
//! while keeping the operator-facing names and environment variables from that
//! ticket, so a reader of the old JSONL can read this one.
//!
//! [`crate::bench`] already times upload/poll/download, but only on the
//! isolated single-shot `bench_one` path. Decay is a property of the *long
//! running* session loop, so it cannot be seen from there.
//!
//! # Enabling
//!
//! ```text
//! QUIP_DRIVER_BUDGET=1                              # off unless exactly "1"
//! QUIP_DRIVER_BUDGET_WINDOW=60                      # report period, seconds
//! QUIP_DRIVER_BUDGET_OUT=/var/log/quip/budget.jsonl # optional JSONL sink
//! ```
//!
//! Disabled is the default and costs one `bool` test per instrumented region:
//! [`DriverBudget::mark`] does not even read the clock when off.
//!
//! # Reading the result
//!
//! Ask which bucket's share *grows* while `att_per_s` falls. `unaccounted`
//! growing is a finding in its own right — it means the real cost centre is
//! outside every region the driver charges, not that the numbers are noisy.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// One accounted region of the driver loop's wall clock.
///
/// Every variant maps to a specific call in the loop, so a growing share names
/// a specific suspect rather than a general area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    /// Reading the ctrl mailbox to learn which slots the kernel finished.
    Poll,
    /// Host-to-device upload of a job's graph into a slot.
    Upload,
    /// Device-to-host download of a finished slot's packed samples.
    Download,
    /// Scoring downloaded spins into `SamplerResult`s (host CPU).
    Score,
    /// Blocking send of a result to the consumer — the backpressure point.
    Consumer,
    /// Deliberate pause while the governor reports foreign GPU contention.
    Throttle,
    /// The idle backoff taken when a scan observed no completion.
    Spin,
}

impl Bucket {
    /// Every bucket, in report order.
    const ALL: [Self; 7] = [
        Self::Poll,
        Self::Upload,
        Self::Download,
        Self::Score,
        Self::Consumer,
        Self::Throttle,
        Self::Spin,
    ];

    /// Dense index into the accumulator array.
    const fn index(self) -> usize {
        match self {
            Self::Poll => 0,
            Self::Upload => 1,
            Self::Download => 2,
            Self::Score => 3,
            Self::Consumer => 4,
            Self::Throttle => 5,
            Self::Spin => 6,
        }
    }

    /// Short operator-facing name, as it appears in the log line and JSONL.
    const fn label(self) -> &'static str {
        match self {
            Self::Poll => "poll",
            Self::Upload => "ul",
            Self::Download => "dl",
            Self::Score => "score",
            Self::Consumer => "consumer",
            Self::Throttle => "throttle",
            Self::Spin => "spin",
        }
    }
}

/// One window's accounting, frozen for reporting.
///
/// Split out from [`DriverBudget`] so the share arithmetic and the log line are
/// testable without waiting on a real window or touching a clock.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// 1-based window number since process start.
    pub window: u64,
    /// Process uptime at the close of this window.
    pub uptime: Duration,
    /// Wall-clock length of this window.
    pub elapsed: Duration,
    /// Jobs completed during this window.
    pub completions: u64,
    /// Time charged to each bucket, indexed by [`Bucket::index`].
    pub charged: [Duration; 7],
}

impl Snapshot {
    /// Completed jobs per second over the window — the `att/s` the decay
    /// reports track. Zero for a zero-length window.
    #[must_use]
    pub fn att_per_s(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        // Completions per window are bounded by the driver's real throughput
        // (single digits per second here), far inside f64's exact integer
        // range, so this cast cannot lose a count.
        #[allow(clippy::cast_precision_loss)]
        let completions = self.completions as f64;
        completions / secs
    }

    /// Fraction of the window charged to `bucket`, in `0.0..=1.0`.
    #[must_use]
    pub fn share(&self, bucket: Bucket) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.charged[bucket.index()].as_secs_f64() / secs
    }

    /// Window time charged to no bucket at all.
    ///
    /// Clamped at zero: the regions are timed independently, so rounding at
    /// the edges can sum a hair over the window, and a negative "unaccounted"
    /// would read as a measurement bug rather than the rounding it is.
    #[must_use]
    pub fn unaccounted_share(&self) -> f64 {
        let accounted: f64 = Bucket::ALL.iter().map(|b| self.share(*b)).sum();
        (1.0 - accounted).max(0.0)
    }

    /// The single-line operator report.
    #[must_use]
    pub fn format_line(&self) -> String {
        let buckets = Bucket::ALL
            .iter()
            .map(|b| format!("{}={:.1}%", b.label(), self.share(*b) * 100.0))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "[QUI-870 budget] win={} up={:.1}min att/s={:.2} | {} unacct={:.1}%",
            self.window,
            self.uptime.as_secs_f64() / 60.0,
            self.att_per_s(),
            buckets,
            self.unaccounted_share() * 100.0,
        )
    }

    /// The same window as a single JSON object, for the JSONL sink.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("win".into(), self.window.into());
        obj.insert("uptime_s".into(), self.uptime.as_secs_f64().into());
        obj.insert("window_s".into(), self.elapsed.as_secs_f64().into());
        obj.insert("completions".into(), self.completions.into());
        obj.insert("att_per_s".into(), self.att_per_s().into());
        for bucket in Bucket::ALL {
            obj.insert(bucket.label().into(), self.share(bucket).into());
        }
        obj.insert("unaccounted".into(), self.unaccounted_share().into());
        serde_json::Value::Object(obj)
    }
}

/// Accumulates driver-loop time and reports it once per window.
///
/// Construct once per process with [`DriverBudget::from_env`] and thread it
/// through the driver loop. When disabled every method is a `bool` test.
#[derive(Debug)]
pub struct DriverBudget {
    enabled: bool,
    window: Duration,
    out: Option<PathBuf>,
    started: Instant,
    window_start: Instant,
    window_index: u64,
    charged: [Duration; 7],
    completions: u64,
}

impl DriverBudget {
    /// Read the QUI-870 environment contract.
    ///
    /// Disabled unless `QUIP_DRIVER_BUDGET` is exactly `1`, so a stray
    /// `QUIP_DRIVER_BUDGET=0` or `=false` cannot silently turn instrumentation
    /// on in production. An unparseable or zero window falls back to 60s
    /// rather than reporting every iteration.
    #[must_use]
    pub fn from_env() -> Self {
        let enabled = std::env::var("QUIP_DRIVER_BUDGET").is_ok_and(|v| v == "1");
        let window = std::env::var("QUIP_DRIVER_BUDGET_WINDOW")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|s| *s > 0)
            .unwrap_or(60);
        let out = std::env::var("QUIP_DRIVER_BUDGET_OUT")
            .ok()
            .filter(|p| !p.is_empty())
            .map(PathBuf::from);
        if enabled {
            tracing::info!(
                window_s = window,
                out = ?out,
                "QUI-870 driver time-budget accounting enabled"
            );
        }
        let now = Instant::now();
        Self {
            enabled,
            window: Duration::from_secs(window),
            out,
            started: now,
            window_start: now,
            window_index: 0,
            charged: [Duration::ZERO; 7],
            completions: 0,
        }
    }

    /// A budget that never measures or reports, for callers with no
    /// instrumentation contract of their own (tests, the batch `sample` path).
    #[must_use]
    pub fn disabled() -> Self {
        let now = Instant::now();
        Self {
            enabled: false,
            window: Duration::from_mins(1),
            out: None,
            started: now,
            window_start: now,
            window_index: 0,
            charged: [Duration::ZERO; 7],
            completions: 0,
        }
    }

    /// Whether this budget is accounting at all.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Open a timed region, or `None` when disabled.
    ///
    /// Reading the clock is the cost of this instrumentation, so a disabled
    /// budget deliberately does not read it.
    #[must_use]
    #[inline]
    pub fn mark(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    /// Charge the time since `mark` to `bucket`. A `None` mark is a no-op, so
    /// call sites need no `if enabled` of their own.
    #[inline]
    pub fn charge(&mut self, bucket: Bucket, mark: Option<Instant>) {
        if let Some(start) = mark {
            self.charged[bucket.index()] += start.elapsed();
        }
    }

    /// Count jobs that produced a result, the numerator of `att/s`.
    #[inline]
    pub fn record_completions(&mut self, n: u64) {
        if self.enabled {
            self.completions += n;
        }
    }

    /// Close and report the window if it has elapsed. Call once per driver
    /// iteration; it is a `bool` test until the window is actually up.
    pub fn maybe_report(&mut self) {
        if !self.enabled || self.window_start.elapsed() < self.window {
            return;
        }
        let snapshot = self.close_window();
        tracing::info!("{}", snapshot.format_line());
        self.append_jsonl(&snapshot);
    }

    /// Freeze the current window and reset the accumulators for the next one.
    fn close_window(&mut self) -> Snapshot {
        self.window_index += 1;
        let snapshot = Snapshot {
            window: self.window_index,
            uptime: self.started.elapsed(),
            elapsed: self.window_start.elapsed(),
            completions: self.completions,
            charged: self.charged,
        };
        self.charged = [Duration::ZERO; 7];
        self.completions = 0;
        self.window_start = Instant::now();
        snapshot
    }

    /// Append one window to the JSONL sink, if one is configured.
    ///
    /// Opened per window rather than held: one open a minute is free, and it
    /// means an external log rotation takes effect on the next window instead
    /// of leaving us writing to an unlinked inode for the rest of a 4h run —
    /// which matters here, because log rotation is what the QUI-922 run under
    /// investigation had just turned on.
    fn append_jsonl(&self, snapshot: &Snapshot) {
        let Some(path) = self.out.as_ref() else {
            return;
        };
        let write = || -> std::io::Result<()> {
            let mut f = OpenOptions::new().create(true).append(true).open(path)?;
            writeln!(f, "{}", snapshot.to_json())
        };
        if let Err(e) = write() {
            // Warn rather than fail the miner: losing the JSONL costs this
            // investigation, but the log line above still carries the window,
            // and mining must not stop because a debug sink is unwritable.
            tracing::warn!(path = ?path, error = %e, "driver budget: JSONL append failed");
        }
    }
}

#[cfg(test)]
mod budget_tests {
    use super::{Bucket, DriverBudget, Snapshot};
    use std::time::Duration;

    fn snapshot(elapsed_s: u64, completions: u64, charged: [Duration; 7]) -> Snapshot {
        Snapshot {
            window: 3,
            uptime: Duration::from_mins(48),
            elapsed: Duration::from_secs(elapsed_s),
            completions,
            charged,
        }
    }

    #[test]
    fn a_disabled_budget_never_reads_the_clock_or_counts() {
        let mut b = DriverBudget::disabled();
        assert!(!b.is_enabled());
        assert!(b.mark().is_none());

        // Charging a `None` mark and counting completions must both no-op, so
        // an instrumented loop is safe to run with the budget off.
        b.charge(Bucket::Poll, None);
        b.record_completions(5);
        let snap = b.close_window();
        assert_eq!(snap.completions, 0);
        assert_eq!(snap.charged, [Duration::ZERO; 7]);
    }

    #[test]
    fn shares_are_fractions_of_the_window_and_att_per_s_is_a_rate() {
        let mut charged = [Duration::ZERO; 7];
        charged[Bucket::Consumer.index()] = Duration::from_secs(30);
        charged[Bucket::Spin.index()] = Duration::from_secs(15);
        let snap = snapshot(60, 156, charged);

        assert!((snap.share(Bucket::Consumer) - 0.5).abs() < 1e-9);
        assert!((snap.share(Bucket::Spin) - 0.25).abs() < 1e-9);
        assert!((snap.share(Bucket::Poll) - 0.0).abs() < 1e-9);
        assert!((snap.att_per_s() - 2.6).abs() < 1e-9);
        // Everything not charged is unaccounted, and it is a finding, not noise.
        assert!((snap.unaccounted_share() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn unaccounted_clamps_instead_of_going_negative() {
        // Independently timed regions can sum just past the window; a negative
        // share would read as a bug in the instrument rather than rounding.
        let mut charged = [Duration::ZERO; 7];
        charged[Bucket::Poll.index()] = Duration::from_secs(61);
        let snap = snapshot(60, 0, charged);

        assert!((snap.unaccounted_share() - 0.0).abs() < 1e-9);
        assert!((snap.att_per_s() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn a_zero_length_window_reports_zero_rather_than_dividing_by_zero() {
        let snap = snapshot(0, 10, [Duration::ZERO; 7]);

        assert!((snap.att_per_s() - 0.0).abs() < 1e-9);
        assert!((snap.share(Bucket::Poll) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn closing_a_window_resets_the_accumulators() {
        let mut b = DriverBudget::disabled();
        b.enabled = true;
        b.record_completions(4);
        b.charge(Bucket::Download, Some(std::time::Instant::now()));

        let first = b.close_window();
        assert_eq!(first.window, 1);
        assert_eq!(first.completions, 4);

        let second = b.close_window();
        assert_eq!(second.window, 2);
        assert_eq!(
            second.completions, 0,
            "counts must not carry across windows"
        );
        assert_eq!(second.charged, [Duration::ZERO; 7]);
    }

    #[test]
    fn the_report_line_carries_every_bucket_and_the_rate() {
        let mut charged = [Duration::ZERO; 7];
        charged[Bucket::Consumer.index()] = Duration::from_secs(30);
        let line = snapshot(60, 156, charged).format_line();

        for bucket in Bucket::ALL {
            assert!(line.contains(bucket.label()), "missing {}", bucket.label());
        }
        assert!(line.contains("att/s=2.60"), "{line}");
        assert!(line.contains("consumer=50.0%"), "{line}");
        assert!(line.contains("unacct="), "{line}");
    }

    #[test]
    fn json_carries_the_same_numbers_as_the_line() {
        let mut charged = [Duration::ZERO; 7];
        charged[Bucket::Throttle.index()] = Duration::from_secs(6);
        let json = snapshot(60, 120, charged).to_json();

        assert_eq!(json["win"], 3);
        assert_eq!(json["completions"], 120);
        assert!((json["att_per_s"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((json["throttle"].as_f64().unwrap() - 0.1).abs() < 1e-9);
        assert!((json["window_s"].as_f64().unwrap() - 60.0).abs() < 1e-9);
    }
}

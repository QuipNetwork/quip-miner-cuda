//! Pure parser for external `nsys`/`ncu` reports, folded into the
//! [`crate::schema`] per-part dataset.
//!
//! The production streaming kernel is one persistent launch that Nsight
//! cannot attribute per-sweep (see `bench.rs`'s module docs for how the
//! isolated single-shot launch path works around that). This module reads
//! the resulting `nsys stats --report cuda_gpu_kern_sum --format csv` and
//! `ncu --csv` reports and folds them into a [`crate::schema::BenchRecord`]:
//! the kernel-launch duration overwrites the device `launch` part, a
//! two-sweep-count derivation fills in `sweep.per_call_ns`, and SM
//! efficiency / achieved occupancy become [`crate::schema::Metric`]s.
//!
//! CSV is hand-rolled (quoted fields, comma-separated, columns located by
//! name so a Nsight version bump that reorders columns fails loudly instead
//! of silently mis-parsing) rather than pulling in the `csv` crate for this
//! small, fixed shape.

use crate::schema::{BenchRecord, Metric, Source};
use thiserror::Error;

/// Failures parsing or folding a Nsight report.
#[derive(Debug, Error)]
pub enum NsightError {
    /// A CSV column the parser needs by name is absent from the header.
    #[error("nsight csv missing column {0:?}")]
    MissingColumn(String),
    /// A cell that should parse as a number did not.
    #[error("nsight csv bad number {0:?}")]
    BadNumber(String),
    /// The report had a header but no data row survived filtering.
    #[error("nsight csv has no kernel row")]
    NoKernelRow,
}

/// One `nsys` `cuda_gpu_kern_sum` row for a self-feeding kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernSum {
    /// Kernel symbol name (`cuda_sa_self_feeding` / `cuda_gibbs_self_feeding`).
    pub name: String,
    /// Number of launches this row summarizes.
    pub instances: u64,
    /// Summed duration across `instances`, nanoseconds.
    pub total_ns: u64,
    /// `total_ns / instances`, as reported by `nsys` (not recomputed here).
    pub avg_ns: u64,
}

/// One `ncu --csv` launch's metrics, after grouping the long-form rows by ID.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NcuMetrics {
    /// `gpu__time_duration.sum`, nanoseconds.
    pub duration_ns: u64,
    /// `sm__throughput.avg.pct_of_peak_sustained_elapsed`, percent.
    pub sm_efficiency_pct: f64,
    /// `sm__warps_active.avg.pct_of_peak_sustained_active`, percent.
    pub achieved_occupancy_pct: f64,
}

/// Split one CSV line into unquoted fields, honoring `""`-escaped quotes.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => fields.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    fields.push(cur);
    fields
}

/// Locate a header column by exact (unquoted) name.
fn find_column(header: &[String], name: &str) -> Result<usize, NsightError> {
    header
        .iter()
        .position(|h| h.trim_matches('"') == name)
        .ok_or_else(|| NsightError::MissingColumn(name.to_owned()))
}

fn field<'a>(fields: &'a [String], idx: usize, column: &str) -> Result<&'a str, NsightError> {
    fields
        .get(idx)
        .map(|s| s.trim_matches('"'))
        .ok_or_else(|| NsightError::MissingColumn(column.to_owned()))
}

fn parse_u64(s: &str) -> Result<u64, NsightError> {
    s.replace(',', "")
        .trim()
        .parse::<u64>()
        .map_err(|_| NsightError::BadNumber(s.to_owned()))
}

fn parse_f64(s: &str) -> Result<f64, NsightError> {
    s.replace(',', "")
        .trim()
        .parse::<f64>()
        .map_err(|_| NsightError::BadNumber(s.to_owned()))
}

/// Parse an `nsys stats --report cuda_gpu_kern_sum --format csv` report,
/// keeping only rows whose kernel name contains `self_feeding` (the two
/// consensus kernels; a report may also list NVRTC-internal or memcpy rows).
///
/// # Errors
///
/// [`NsightError::MissingColumn`] if the header lacks `Name`, `Instances`,
/// `Total Time (ns)`, or `Avg (ns)`; [`NsightError::BadNumber`] if a numeric
/// cell does not parse; [`NsightError::NoKernelRow`] if no row survives.
pub fn parse_nsys_kern_sum(csv: &str) -> Result<Vec<KernSum>, NsightError> {
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header = split_csv_line(lines.next().ok_or(NsightError::NoKernelRow)?);
    let name_i = find_column(&header, "Name")?;
    let total_i = find_column(&header, "Total Time (ns)")?;
    let inst_i = find_column(&header, "Instances")?;
    let avg_i = find_column(&header, "Avg (ns)")?;

    let mut out = Vec::new();
    for line in lines {
        let fields = split_csv_line(line);
        let name = field(&fields, name_i, "Name")?.to_owned();
        if !name.contains("self_feeding") {
            continue;
        }
        out.push(KernSum {
            total_ns: parse_u64(field(&fields, total_i, "Total Time (ns)")?)?,
            instances: parse_u64(field(&fields, inst_i, "Instances")?)?,
            avg_ns: parse_u64(field(&fields, avg_i, "Avg (ns)")?)?,
            name,
        });
    }
    if out.is_empty() {
        return Err(NsightError::NoKernelRow);
    }
    Ok(out)
}

/// `(launch id, duration, sm-efficiency, achieved-occupancy)`, built up as
/// the tracked metrics are seen for that launch in [`parse_ncu_csv`].
type NcuRow = (String, Option<f64>, Option<f64>, Option<f64>);

const METRIC_DURATION: &str = "gpu__time_duration.sum";
const METRIC_SM_EFFICIENCY: &str = "sm__throughput.avg.pct_of_peak_sustained_elapsed";
const METRIC_OCCUPANCY: &str = "sm__warps_active.avg.pct_of_peak_sustained_active";

/// Parse an `ncu --csv` long-form report (one row per metric per launch),
/// grouping by the `ID` column into one [`NcuMetrics`] per launch. Launch IDs
/// whose rows carry none of the three tracked metrics are dropped.
///
/// # Errors
///
/// [`NsightError::MissingColumn`] if the header lacks `ID`, `Metric Name`, or
/// `Metric Value`; [`NsightError::BadNumber`] if a metric value does not
/// parse; [`NsightError::NoKernelRow`] if no launch carries a tracked metric.
pub fn parse_ncu_csv(csv: &str) -> Result<Vec<NcuMetrics>, NsightError> {
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header = split_csv_line(lines.next().ok_or(NsightError::NoKernelRow)?);
    let id_i = find_column(&header, "ID")?;
    let metric_name_i = find_column(&header, "Metric Name")?;
    let metric_value_i = find_column(&header, "Metric Value")?;

    let mut by_id: Vec<NcuRow> = Vec::new();
    for line in lines {
        let fields = split_csv_line(line);
        let id = field(&fields, id_i, "ID")?.to_owned();
        let metric_name = field(&fields, metric_name_i, "Metric Name")?.to_owned();
        let value = parse_f64(field(&fields, metric_value_i, "Metric Value")?)?;
        let entry = if let Some(e) = by_id.iter_mut().find(|(existing, ..)| *existing == id) {
            e
        } else {
            by_id.push((id, None, None, None));
            let last = by_id.len() - 1;
            &mut by_id[last]
        };
        match metric_name.as_str() {
            METRIC_DURATION => entry.1 = Some(value),
            METRIC_SM_EFFICIENCY => entry.2 = Some(value),
            METRIC_OCCUPANCY => entry.3 = Some(value),
            _ => {}
        }
    }

    let out: Vec<NcuMetrics> = by_id
        .into_iter()
        .filter_map(|(_, dur, eff, occ)| {
            if dur.is_none() && eff.is_none() && occ.is_none() {
                return None;
            }
            // Round rather than truncate: ncu reports duration as an exact
            // float nanosecond count in practice, but round defends against
            // any sub-ns fractional noise in the source report.
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let duration_ns = dur.map_or(0, |d| d.max(0.0).round() as u64);
            Some(NcuMetrics {
                duration_ns,
                sm_efficiency_pct: eff.unwrap_or(0.0),
                achieved_occupancy_pct: occ.unwrap_or(0.0),
            })
        })
        .collect();
    if out.is_empty() {
        return Err(NsightError::NoKernelRow);
    }
    Ok(out)
}

/// Two-point per-sweep derivation: `nsys` sees a whole kernel launch, not
/// individual sweeps, so the sweep-variable cost is isolated by differencing
/// two runs of the same model at different sweep counts — the constant
/// init+pack overhead cancels: `(dur_hi - dur_lo) / (s_hi - s_lo)`.
///
/// Returns `0` (rather than panicking or dividing by zero) if the two points
/// do not actually bracket a positive slope — a malformed pair of runs should
/// report "no signal", not a bogus or wrapped value.
#[must_use]
pub fn derive_per_sweep(dur_lo_ns: u64, s_total_lo: u64, dur_hi_ns: u64, s_total_hi: u64) -> u64 {
    if s_total_hi <= s_total_lo || dur_hi_ns <= dur_lo_ns {
        return 0;
    }
    (dur_hi_ns - dur_lo_ns) / (s_total_hi - s_total_lo)
}

/// Fold a Nsight report into `record`: the measured kernel-sum duration
/// overwrites the device `launch` part, a derived per-sweep cost (if
/// supplied) overwrites the `sweep` part, and `ncu` metrics (if supplied) are
/// appended to `record.metrics`. Does not call `record.finalize()`: these are
/// all device-scope changes plus non-time metrics, none of which enter
/// `residual_ns` (a host-only reconciliation) — the caller only needs to
/// re-finalize if it also touched a host part.
pub fn fold_into(
    record: &mut BenchRecord,
    kern: &[KernSum],
    ncu: Option<&[NcuMetrics]>,
    per_sweep_ns: Option<u64>,
) {
    if let Some(row) = kern.first() {
        if let Some(launch) = record.parts.iter_mut().find(|p| p.part == "launch") {
            launch.total_ns = row.avg_ns;
            launch.source = Source::Nsys;
            #[allow(clippy::cast_precision_loss)] // launch is a single-call part; count == 1
            let per_call_ns = row.avg_ns as f64 / launch.count.max(1) as f64;
            launch.per_call_ns = per_call_ns;
        }
    }
    if let Some(per_sweep) = per_sweep_ns {
        if let Some(sweep) = record.parts.iter_mut().find(|p| p.part == "sweep") {
            #[allow(clippy::cast_precision_loss)] // per-sweep ns stays far below 2^53
            let per_call_ns = per_sweep as f64;
            sweep.per_call_ns = per_call_ns;
            sweep.total_ns = per_sweep.saturating_mul(sweep.count);
            sweep.source = Source::Nsys;
        }
    }
    if let Some(rows) = ncu.and_then(<[NcuMetrics]>::first) {
        record.metrics.push(Metric::new(
            "sm_efficiency",
            rows.sm_efficiency_pct,
            "%",
            Source::Ncu,
        ));
        record.metrics.push(Metric::new(
            "achieved_occupancy",
            rows.achieved_occupancy_pct,
            "%",
            Source::Ncu,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::{derive_per_sweep, parse_ncu_csv, parse_nsys_kern_sum};

    const KERN_LO: &str = "\"Time (%)\",\"Total Time (ns)\",\"Instances\",\"Avg (ns)\",\"Med (ns)\",\"Min (ns)\",\"Max (ns)\",\"StdDev (ns)\",\"Name\"\n\"100.0\",\"4000000\",\"5\",\"800000\",\"800000\",\"790000\",\"815000\",\"9000\",\"cuda_sa_self_feeding\"\n";
    const KERN_HI: &str = "\"Time (%)\",\"Total Time (ns)\",\"Instances\",\"Avg (ns)\",\"Med (ns)\",\"Min (ns)\",\"Max (ns)\",\"StdDev (ns)\",\"Name\"\n\"100.0\",\"28000000\",\"5\",\"5600000\",\"5600000\",\"5580000\",\"5630000\",\"20000\",\"cuda_sa_self_feeding\"\n";
    const NCU: &str = "\"ID\",\"Kernel Name\",\"Metric Name\",\"Metric Unit\",\"Metric Value\"\n\"0\",\"cuda_sa_self_feeding\",\"gpu__time_duration.sum\",\"ns\",\"812000.00\"\n\"0\",\"cuda_sa_self_feeding\",\"sm__throughput.avg.pct_of_peak_sustained_elapsed\",\"%\",\"76.30\"\n\"0\",\"cuda_sa_self_feeding\",\"sm__warps_active.avg.pct_of_peak_sustained_active\",\"%\",\"61.10\"\n";

    #[test]
    fn parses_nsys_kern_sum_row() {
        let rows = parse_nsys_kern_sum(KERN_LO).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "cuda_sa_self_feeding");
        assert_eq!(rows[0].instances, 5);
        assert_eq!(rows[0].total_ns, 4_000_000);
        assert_eq!(rows[0].avg_ns, 800_000);
    }

    #[test]
    fn derives_per_sweep_by_cancelling_fixed_overhead() {
        let lo = parse_nsys_kern_sum(KERN_LO).unwrap()[0].avg_ns;
        let hi = parse_nsys_kern_sum(KERN_HI).unwrap()[0].avg_ns;
        let per_sweep = derive_per_sweep(lo, 1024, hi, 8192);
        assert_eq!(per_sweep, 669);
    }

    #[test]
    fn derive_per_sweep_returns_zero_for_non_bracketing_points() {
        assert_eq!(derive_per_sweep(100, 10, 100, 10), 0, "no sweep delta");
        assert_eq!(derive_per_sweep(100, 20, 200, 10), 0, "s_hi <= s_lo");
        assert_eq!(derive_per_sweep(200, 10, 100, 20), 0, "dur_hi <= dur_lo");
    }

    #[test]
    fn parses_ncu_metrics() {
        let m = parse_ncu_csv(NCU).unwrap();
        assert_eq!(m.len(), 1);
        assert!(m[0].sm_efficiency_pct > 0.0);
        assert!(m[0].achieved_occupancy_pct > 0.0);
        assert!(m[0].duration_ns > 0);
    }

    #[test]
    fn missing_column_is_reported_by_name() {
        let bad = "\"Foo\",\"Bar\"\n\"1\",\"2\"\n";
        let err = parse_nsys_kern_sum(bad).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Name") || msg.contains("Total Time"));
    }
}

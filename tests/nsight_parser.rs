//! Always-run parser tests over checked-in Nsight CSV fixtures — no GPU
//! needed. Confirms `nsys`/`ncu` CSV parsing, the two-point per-sweep
//! derivation, and folding a parsed report into a [`BenchRecord`] the
//! `bench run` host path already emitted.

use quip_miner_cuda::bench::{assemble_record, CellConfig, HostSpans};
use quip_miner_cuda::nsight::{derive_per_sweep, fold_into, parse_ncu_csv, parse_nsys_kern_sum};
use quip_miner_cuda::schema::{Scope, Source};
use quip_miner_cuda::streaming::DeviceTimings;

const KERN_LO: &str = include_str!("fixtures/nsight/kern_sum_lo.csv");
const KERN_HI: &str = include_str!("fixtures/nsight/kern_sum_hi.csv");
const NCU: &str = include_str!("fixtures/nsight/ncu_metrics.csv");

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
    // lo: S_total=1024, avg 800_000 ns; hi: S_total=8192, avg 5_600_000 ns.
    // slope = (5_600_000 - 800_000) / (8192 - 1024) = 4_800_000 / 7168 = 669.
    let lo = parse_nsys_kern_sum(KERN_LO).unwrap()[0].avg_ns;
    let hi = parse_nsys_kern_sum(KERN_HI).unwrap()[0].avg_ns;
    let per_sweep = derive_per_sweep(lo, 1024, hi, 8192);
    assert_eq!(per_sweep, 669);
}

#[test]
fn parses_ncu_metrics() {
    let m = parse_ncu_csv(NCU).unwrap();
    assert_eq!(m.len(), 1);
    assert!(m[0].sm_efficiency_pct > 0.0);
    assert!(m[0].achieved_occupancy_pct > 0.0);
    assert!(m[0].duration_ns > 0);
}

/// Build a `BenchRecord` the way `bench run`'s host path would, using purely
/// synthetic (no-GPU) device/host timings.
fn synthetic_record() -> quip_miner_cuda::schema::BenchRecord {
    let dev = DeviceTimings {
        kernel_ns: 750_000, // pre-fold placeholder; fold_into overwrites this
        upload_ns: 100_000,
        download_ns: 50_000,
        poll_wait_ns: 5_000,
    };
    let cfg = CellConfig {
        reads: 8,
        num_sweeps: 1024,
        sweeps_per_beta: 4,
        num_betas: 256, // S_total = num_betas * sweeps_per_beta = 1024
        nodes: 512,
        edges: 512,
    };
    let host = HostSpans {
        problem_setup_ns: 10_000,
        beta_build_ns: 2_000,
        energy_score_ns: 40_000,
        jit_ns: None,
    };
    assemble_record(
        "cuda-sa", "TestGPU", "0xnonce", "0xtopo", &cfg, &dev, &host, 900_000,
    )
}

#[test]
fn fold_round_trip_overwrites_launch_and_sweep_and_appends_metrics() {
    let mut record = synthetic_record();
    let kern_rows = parse_nsys_kern_sum(KERN_LO).unwrap();
    let lo_avg = kern_rows[0].avg_ns;
    let hi_avg = parse_nsys_kern_sum(KERN_HI).unwrap()[0].avg_ns;
    let per_sweep = derive_per_sweep(lo_avg, 1024, hi_avg, 8192);
    let ncu_rows = parse_ncu_csv(NCU).unwrap();

    fold_into(&mut record, &kern_rows, Some(&ncu_rows), Some(per_sweep));

    let launch = record.parts.iter().find(|p| p.part == "launch").unwrap();
    assert_eq!(launch.scope, Scope::Device);
    assert_eq!(
        launch.total_ns, 800_000,
        "launch overwritten from nsys kern-sum avg"
    );
    assert_eq!(launch.source, Source::Nsys);

    let sweep = record.parts.iter().find(|p| p.part == "sweep").unwrap();
    assert!((sweep.per_call_ns - 669.0).abs() < f64::EPSILON);
    assert_eq!(sweep.source, Source::Nsys);

    assert_eq!(
        record.metrics.len(),
        2,
        "sm_efficiency + achieved_occupancy appended"
    );
    let sm = record
        .metrics
        .iter()
        .find(|m| m.name == "sm_efficiency")
        .unwrap();
    assert_eq!(sm.unit, "%");
    assert_eq!(sm.source, Source::Ncu);
    assert!(record
        .metrics
        .iter()
        .any(|m| m.name == "achieved_occupancy"));

    // parts[] stays time-only: folding metrics never adds a metrics-shaped part.
    assert!(!record.parts.iter().any(|p| p.part.contains("efficiency")));
    assert!(!record.parts.iter().any(|p| p.part.contains("occupancy")));
}

#[test]
fn fold_into_without_ncu_or_per_sweep_leaves_those_untouched() {
    let mut record = synthetic_record();
    let original_sweep_per_call = record
        .parts
        .iter()
        .find(|p| p.part == "sweep")
        .unwrap()
        .per_call_ns;
    let kern_rows = parse_nsys_kern_sum(KERN_LO).unwrap();

    fold_into(&mut record, &kern_rows, None, None);

    let launch = record.parts.iter().find(|p| p.part == "launch").unwrap();
    assert_eq!(
        launch.total_ns, 800_000,
        "launch still overwritten (kern rows given)"
    );
    let sweep = record.parts.iter().find(|p| p.part == "sweep").unwrap();
    assert!(
        (sweep.per_call_ns - original_sweep_per_call).abs() < f64::EPSILON,
        "no per_sweep_ns supplied -> sweep part untouched"
    );
    assert!(
        record.metrics.is_empty(),
        "no ncu rows supplied -> no metrics appended"
    );
}

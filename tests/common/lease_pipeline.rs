//! Fixed-wall-time measurements through the production MSA binary and session.

use super::{ensure_built, lease, permissive_target, profile_bin, target, Session};
use quip_solver_core::quip_proto::v1::{coord_msg, Cancel, EdgeList, Topology};
use quip_solver_core::quip_protocol::lease::{verify_lease_result, TopologyView};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Write;
use std::time::Duration;
use tokio::time::Instant;

const HASH: [u8; 32] = [0x5b; 32];
const SWEEPS: u32 = 29_568;
const READS: u32 = 128;

#[derive(Deserialize)]
struct Fixture {
    nodes: Vec<u32>,
    edges: Vec<(u32, u32)>,
    allowed_h_milli: Vec<i32>,
}

fn topology() -> Topology {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/aglais.spec.json")).unwrap();
    assert_eq!(fixture.nodes.len(), 4577);
    assert_eq!(fixture.edges.len(), 41_514);
    Topology {
        hash: HASH.to_vec(),
        nodes: fixture.nodes,
        edges: Some(EdgeList {
            u: fixture.edges.iter().map(|&(u, _)| u).collect(),
            v: fixture.edges.iter().map(|&(_, v)| v).collect(),
        }),
        allowed_h_milli: fixture.allowed_h_milli,
        allowed_j_milli: vec![-1000, 1000],
    }
}

// Use the nearest-rank definition, including all ties at the chosen energy.
fn corpus_target(path: &std::path::Path) -> i64 {
    let csv = std::fs::read_to_string(path).expect("calibration corpus");
    let mut lines = csv.lines();
    let column = lines
        .next()
        .unwrap()
        .split(',')
        .position(|name| name == "full_a")
        .expect("full_a column");
    let mut energies: Vec<i64> = lines
        .map(|line| line.split(',').nth(column).unwrap().parse().unwrap())
        .collect();
    assert_eq!(energies.len(), 4096);
    energies.sort_unstable();
    energies[energies.len().div_ceil(100) - 1]
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "600-second device-0 session benchmark per mode; set LEASE_PIPELINE_OUT"]
#[allow(clippy::too_many_lines)] // Keep setup, timed windows and cancellation in one session lifecycle.
async fn whole_pipeline() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok("0"));
    let out = std::path::PathBuf::from(std::env::var("LEASE_PIPELINE_OUT").expect("evidence path"));
    std::fs::create_dir_all(&out).unwrap();
    let seconds: u64 = std::env::var("LEASE_PIPELINE_SECONDS")
        .unwrap_or_else(|_| "600".into())
        .parse()
        .unwrap();
    assert!(seconds > 0);
    let mode = std::env::var("LEASE_PIPELINE_MODE").unwrap_or_else(|_| "screened".into());
    assert!(mode == "screened" || mode == "direct");
    let threshold = corpus_target(&out.join("corpus-4096.csv"));
    let config = if mode == "direct" {
        "[screen]\nenabled = false"
    } else {
        ""
    };
    ensure_built(&["quip-cuda-msa"]);
    let mut s = Session::start(&profile_bin("quip-cuda-msa"), 4, config).await;
    let topo = topology();
    let view = TopologyView::from_proto(&topo).unwrap();
    s.send(coord_msg::Msg::Topology(topo)).await;
    let mut target = target(threshold, SWEEPS);
    target.num_reads = READS;
    s.send(coord_msg::Msg::SetTarget(target)).await;
    let mut job = lease(b"pipeline", 1, HASH, 0, 10_000_000);
    // Match the corpus's nonce domain and prefix for both modes.
    job.generator.as_mut().unwrap().base_salt = vec![0x33; 32];
    let spec = job.generator.clone().unwrap();
    let started = Instant::now();
    s.send(coord_msg::Msg::Job(job)).await;
    let mut series = Vec::new();
    let mut rows = std::fs::File::create(out.join(format!("pipeline-{mode}-pushes.csv"))).unwrap();
    writeln!(rows, "window,salt_index,best_energy_milli").unwrap();
    let mut seen = HashSet::new();
    let mut pushes = 0u64;
    let mut hits = 0u64;
    let mut best = i64::MAX;
    let mut previous_seconds = 0.0;
    let mut previous_salts = 0;
    let mut previous_pushes = 0;
    let mut previous_hits = 0;
    let mut end = 60.min(seconds);
    loop {
        s.collect_until(started + Duration::from_secs(end)).await;
        let salts = s.jobs_done().await;
        let elapsed = started.elapsed().as_secs_f64();
        let mut window_best = i64::MAX;
        for result in s.log.results.drain(..) {
            let meta = result.meta.as_ref().expect("resolved parameters");
            assert_eq!((meta.reads, meta.sweeps), (READS, SWEEPS));
            assert_eq!(result.solutions.len(), usize::try_from(READS).unwrap());
            let index = u64::from_le_bytes(result.salt[..8].try_into().unwrap());
            assert!(seen.insert(index), "duplicate pushed salt {index}");
            let energy = result
                .solutions
                .iter()
                .map(|s| s.energy_milli)
                .min()
                .unwrap();
            // Check a complete result independently in each window.
            if window_best == i64::MAX {
                verify_lease_result(&spec, &view, &permissive_target(), &result)
                    .expect("independent nonce, coefficients and energy verification");
            }
            window_best = window_best.min(energy);
            pushes += 1;
            hits += u64::from(energy <= threshold);
            best = best.min(energy);
            writeln!(rows, "{end},{index},{energy}").unwrap();
        }
        let entry = json!({
            "nominal_end_seconds": end, "elapsed_seconds": elapsed,
            "window_seconds": elapsed - previous_seconds,
            "salts_completed": salts - previous_salts,
            "pushes": pushes - previous_pushes, "target_hits": hits - previous_hits,
            "best_energy_milli": (window_best != i64::MAX).then_some(window_best),
            "cumulative_salts": salts, "cumulative_pushes": pushes,
            "cumulative_hits": hits,
        });
        println!("pipeline {mode}: {entry}");
        series.push(entry);
        previous_seconds = elapsed;
        previous_salts = salts;
        previous_pushes = pushes;
        previous_hits = hits;
        if end == seconds {
            break;
        }
        end = (end + 60).min(seconds);
    }
    // Keep the cancellation tail separate from the measured windows.
    s.send(coord_msg::Msg::Cancel(Cancel { max_generation: 1 }))
        .await;
    s.until("benchmark cancellation", Duration::from_secs(90), |log| {
        log.done(b"pipeline").is_some() && log.refunds() >= 1
    })
    .await;
    let done = s.log.done(b"pipeline").unwrap().clone();
    let tail_pushes = s.log.results.len();
    assert_eq!(s.shutdown(2000).await, 0);
    let stderr = s.stderr();
    assert!(stderr.contains(if mode == "direct" {
        "Direct"
    } else {
        "Screened"
    }));
    assert!(stderr.contains("CUDA lease finished"));
    let report: Value = json!({
        "mode": mode, "backend_toml": config, "target_milli": threshold,
        "percentile_method": "nearest rank: sorted full_a[ceil(N/100)-1]",
        "nominal_seconds": seconds, "measured_seconds": previous_seconds,
        "resolved_reads": READS, "resolved_sweeps": SWEEPS,
        "shape_source": "SetTarget explicit read and sweep overrides, asserted for every push",
        "salts_completed": previous_salts, "pushes": pushes, "target_hits": hits,
        "best_energy_milli": (best != i64::MAX).then_some(best),
        "cancel_tail_pushes": tail_pushes, "final_salts_done": done.salts_done,
        "commit": std::env::var("PIPELINE_COMMIT").unwrap_or_default(),
        "series": series,
    });
    std::fs::write(
        out.join(format!("pipeline-{mode}.json")),
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
    std::fs::write(out.join(format!("pipeline-{mode}-miner.log")), stderr).unwrap();
    assert!(pushes > 0, "benchmark produced no full results");
    assert!(s.log.fatal.is_none());
    assert!(s.log.rejects.is_empty());
}

//! Schema round-trip guard (no GPU): a `BenchRecord` built the way `bench
//! run`'s host path builds one survives a `serde_json` round trip, and every
//! `scope`/`source` serializes to the contract string the isingmark C plan
//! and Python viz key on. Catches a future rename before it breaks C.

use quip_miner_cuda::bench::{assemble_record, CellConfig, HostSpans};
use quip_miner_cuda::schema::{BenchRecord, Scope, Source};
use quip_miner_cuda::streaming::DeviceTimings;

fn sample_record() -> BenchRecord {
    let dev = DeviceTimings {
        kernel_ns: 800_000,
        upload_ns: 120_000,
        download_ns: 60_000,
        poll_wait_ns: 5_000,
    };
    let cfg = CellConfig {
        reads: 8,
        num_sweeps: 1024,
        sweeps_per_beta: 4,
        num_betas: 256,
        nodes: 64,
        edges: 64,
    };
    let host = HostSpans {
        problem_setup_ns: 10_000,
        beta_build_ns: 2_000,
        energy_score_ns: 40_000,
        jit_ns: Some(500_000),
    };
    assemble_record(
        "cuda-sa", "TestGPU", "0x01", "0xaa", &cfg, &dev, &host, 1_000_000,
    )
}

#[test]
fn bench_record_round_trips_through_json() {
    let rec = sample_record();
    let json = serde_json::to_string(&rec).unwrap();
    let back: BenchRecord = serde_json::from_str(&json).unwrap();

    assert_eq!(back.backend, rec.backend);
    assert_eq!(back.device, rec.device);
    assert_eq!(back.nonce, rec.nonce);
    assert_eq!(back.topology_hash, rec.topology_hash);
    assert_eq!(back.reads, rec.reads);
    assert_eq!(back.sweeps, rec.sweeps);
    assert_eq!(back.sweeps_per_beta, rec.sweeps_per_beta);
    assert_eq!(back.nodes, rec.nodes);
    assert_eq!(back.edges, rec.edges);
    assert_eq!(back.model_total_ns, rec.model_total_ns);
    assert_eq!(back.residual_ns, rec.residual_ns);
    assert_eq!(back.parts.len(), rec.parts.len());
    for (a, b) in rec.parts.iter().zip(back.parts.iter()) {
        assert_eq!(a.part, b.part);
        assert_eq!(a.scope, b.scope);
        assert_eq!(a.total_ns, b.total_ns);
        assert_eq!(a.count, b.count);
        assert!((a.per_call_ns - b.per_call_ns).abs() < 1e-9);
        assert_eq!(a.source, b.source);
    }
}

#[test]
fn scope_and_source_serialize_to_the_locked_contract_strings() {
    assert_eq!(serde_json::to_string(&Scope::Host).unwrap(), "\"host\"");
    assert_eq!(serde_json::to_string(&Scope::Device).unwrap(), "\"device\"");
    assert_eq!(
        serde_json::to_string(&Source::Tracing).unwrap(),
        "\"tracing\""
    );
    assert_eq!(
        serde_json::to_string(&Source::CounterDerived).unwrap(),
        "\"counter-derived\""
    );
    assert_eq!(
        serde_json::to_string(&Source::CudaEvent).unwrap(),
        "\"cuda-event\""
    );
    assert_eq!(serde_json::to_string(&Source::Nsys).unwrap(), "\"nsys\"");
    assert_eq!(serde_json::to_string(&Source::Ncu).unwrap(), "\"ncu\"");
}

#[test]
fn json_object_keys_match_the_locked_unified_record_shape() {
    let rec = sample_record();
    let json = serde_json::to_string(&rec).unwrap();
    for key in [
        "backend",
        "device",
        "nonce",
        "topology_hash",
        "reads",
        "sweeps",
        "sweeps_per_beta",
        "nodes",
        "edges",
        "model_total_ns",
        "parts",
        "residual_ns",
    ] {
        assert!(
            json.contains(&format!("\"{key}\"")),
            "missing key {key:?} in {json}"
        );
    }
    for key in [
        "part",
        "scope",
        "total_ns",
        "count",
        "per_call_ns",
        "source",
    ] {
        assert!(
            json.contains(&format!("\"{key}\"")),
            "missing part-field {key:?}"
        );
    }
}

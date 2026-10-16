//! Probe draws, energies, thresholds, partial batches, and stops on device 0.

use quip_solver_core::TopologyView;
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::screen::{
    reference_energy, reference_masks, Screen, ScreenDomain, ScreenParams,
};
use quip_solver_cuda::KernelKind;
use serde::Deserialize;
use serial_test::serial;
use std::cell::Cell;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct TopologyFixture {
    nodes: Vec<u32>,
    edges: Vec<(u32, u32)>,
    allowed_h_milli: Vec<i32>,
}

fn wire_topology(couplings: &[i32]) -> quip_proto::v1::Topology {
    // Derived from the coordinator's Aglais preset without reordering edges.
    let fixture: TopologyFixture =
        serde_json::from_str(include_str!("fixtures/aglais.spec.json")).unwrap();
    assert_eq!(fixture.nodes.len(), 4577);
    assert_eq!(fixture.edges.len(), 41_514);
    quip_proto::v1::Topology {
        nodes: fixture.nodes,
        edges: Some(quip_proto::v1::EdgeList {
            u: fixture.edges.iter().map(|&(u, _)| u).collect(),
            v: fixture.edges.iter().map(|&(_, v)| v).collect(),
        }),
        allowed_h_milli: fixture.allowed_h_milli,
        allowed_j_milli: couplings.to_vec(),
        ..Default::default()
    }
}

fn topology(couplings: &[i32]) -> TopologyView {
    TopologyView::from_proto(&wire_topology(couplings)).unwrap()
}

#[test]
fn duplicate_native_nodes_are_rejected_before_screen_admission() {
    let mut wire = wire_topology(&[-1000, 1000]);
    wire.nodes.push(wire.nodes[0]);
    assert_eq!(
        TopologyView::from_proto(&wire).unwrap_err(),
        quip_protocol::lease::TopologyError::DuplicateNode(wire.nodes[0])
    );
}

fn device() -> &'static CudaDevice {
    static DEVICE: OnceLock<CudaDevice> = OnceLock::new();
    DEVICE.get_or_init(|| {
        CudaDevice::open_with_nodes(0, KernelKind::Msa, 4577)
            .unwrap_or_else(|error| panic!("CUDA device 0: {error}"))
    })
}

fn nonces(count: usize) -> Vec<[u8; 32]> {
    (0..count)
        .map(|i| {
            let mut nonce = [0xa5; 32];
            nonce[..8].copy_from_slice(&u64::try_from(i + 1).unwrap().to_le_bytes());
            nonce
        })
        .collect()
}

fn verify(reads_log2: u32, couplings: &[i32], threads: u32) {
    let topo = topology(couplings);
    let params = ScreenParams {
        reads_log2,
        threads,
        ..ScreenParams::default()
    };
    let screen = Screen::open(
        device(),
        ScreenDomain::check(device(), &topo, params).unwrap(),
    )
    .unwrap();
    let npg = 32usize >> reads_log2;
    let nonces = nonces(npg + 1);
    let debug = screen.run_debug(&nonces[..npg], 7).unwrap();
    let want = reference_masks(&nonces[..npg], reads_log2, &topo).unwrap();
    assert_eq!(debug.masks.len(), topo.edges.len());
    assert_eq!(debug.state.len(), topo.num_nodes);
    assert_eq!(debug.lanes.len(), 32);
    for (edge, (&got, &expected)) in debug.masks.iter().zip(&want).enumerate() {
        assert_eq!(got, expected, "coupling mask at edge {edge}");
    }
    for (lane, &energy) in debug.lanes.iter().enumerate() {
        let expected = reference_energy(nonces[lane % npg], &debug.state, lane, &topo).unwrap();
        assert_eq!(
            i64::from(energy) * 1000,
            expected,
            "milli energy at lane {lane}"
        );
    }
    for nonce in 0..npg {
        let best = debug
            .lanes
            .iter()
            .skip(nonce)
            .step_by(npg)
            .copied()
            .min()
            .unwrap();
        assert_eq!(
            debug.best[nonce],
            i64::from(best) * 1000,
            "minimum for nonce {nonce}"
        );
    }
    // One full group followed by one real nonce in an otherwise padded group.
    let mut expected = debug.best;
    let tail = screen.run_debug(&nonces[npg..], 7).unwrap();
    let exact_tail = (0..32)
        .step_by(npg)
        .map(|lane| reference_energy(nonces[npg], &tail.state, lane, &topo).unwrap())
        .min()
        .unwrap();
    assert_eq!(tail.best, vec![exact_tail]);
    expected.push(exact_tail);
    let start = std::time::Instant::now();
    let run = screen.run(&nonces, &|| false).unwrap();
    eprintln!(
        "probe reads_log2={reads_log2} threads={threads}: two one-group launches {:?}",
        start.elapsed()
    );
    assert!(!run.stopped);
    assert_eq!(run.best.len(), npg + 1, "padded slots are excluded");
    assert_eq!(
        run.best, expected,
        "input order and exact milli minima across batches"
    );
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU"]
fn thresholds_at_32_threads_match_64_threads() {
    let topo = topology(&[-1000, 1000]);
    let thresholds = |threads| {
        let params = ScreenParams {
            threads,
            sweeps: 2,
            ..ScreenParams::default()
        };
        let shape = ScreenDomain::check(device(), &topo, params).unwrap();
        Screen::open(device(), shape)
            .unwrap()
            .run_debug(&[[0xa5; 32]], 7)
            .unwrap()
            .cut
    };
    let full = thresholds(64);
    assert_eq!(full.len(), 64);
    assert_eq!(full[0], u64::MAX);
    // Aglais max degree 20 gives hot beta ln(2)/40, narrowed to f32 before upload.
    assert_eq!(full[32], 6_085_156_164_771_500_032);
    assert_eq!(thresholds(32), full);
}

#[test]
#[serial]
#[ignore = "requires CUDA GPU"]
fn stop_returns_only_the_completed_prefix() {
    let topo = topology(&[-1000, 1000]);
    let params = ScreenParams {
        sweeps: 8,
        ..ScreenParams::default()
    };
    let shape = ScreenDomain::check(device(), &topo, params).unwrap();
    let screen = Screen::open(device(), shape).unwrap();
    let nonces = nonces(80);
    let before = screen.run(&nonces, &|| true).unwrap();
    assert!(before.stopped);
    assert!(before.best.is_empty());
    let calls = Cell::new(0);
    let stop = || {
        calls.set(calls.get() + 1);
        calls.get() == 2
    };
    let expected = screen.run_debug(&nonces[..8], 7).unwrap().best;
    let run = screen.run(&nonces, &stop).unwrap();
    assert!(run.stopped);
    assert_eq!(calls.get(), 2);
    assert_eq!(run.best, expected);
    assert_eq!(run.best.len(), 8);
    let empty = screen
        .run(&[], &|| panic!("empty input must not launch"))
        .unwrap();
    assert!(!empty.stopped);
    assert!(empty.best.is_empty());
    // The same screen remains usable after a partial result.
    assert_eq!(screen.run(&nonces[..8], &|| false).unwrap().best, expected);
}

macro_rules! protocol_checks {
    ($($name:ident: ($reads_log2:expr, $couplings:expr, $threads:expr)),+ $(,)?) => {
        $(
            #[test]
            #[serial]
            #[ignore = "requires CUDA GPU; run with CUDA_VISIBLE_DEVICES=0 cargo test --release --test screen_gpu -- --ignored"]
            fn $name() {
                verify($reads_log2, &$couplings, $threads);
            }
        )+
    };
}

protocol_checks! {
    reads4_negative_first_threads256: (2, [-1000, 1000], 256),
    reads4_positive_first_threads256: (2, [1000, -1000], 256),
    reads8_negative_first_threads256: (3, [-1000, 1000], 256),
    reads8_positive_first_threads256: (3, [1000, -1000], 256),
    reads16_negative_first_threads256: (4, [-1000, 1000], 256),
    reads16_positive_first_threads256: (4, [1000, -1000], 256),
    reads32_negative_first_threads256: (5, [-1000, 1000], 256),
    reads32_positive_first_threads256: (5, [1000, -1000], 256),
    reads4_negative_first_threads32: (2, [-1000, 1000], 32),
    reads4_positive_first_threads32: (2, [1000, -1000], 32),
    reads8_negative_first_threads32: (3, [-1000, 1000], 32),
    reads8_positive_first_threads32: (3, [1000, -1000], 32),
    reads16_negative_first_threads32: (4, [-1000, 1000], 32),
    reads16_positive_first_threads32: (4, [1000, -1000], 32),
    reads32_negative_first_threads32: (5, [-1000, 1000], 32),
    reads32_positive_first_threads32: (5, [1000, -1000], 32),
}

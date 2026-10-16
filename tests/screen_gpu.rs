//! Probe coupling draws and lane energies against the protocol on device 0.

use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::screen::{
    reference_energy, reference_masks, Screen, ScreenParams, ScreenTopology,
};
use quip_solver_cuda::KernelKind;
use serde::Deserialize;
use serial_test::serial;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct TopologyFixture {
    nodes: Vec<u32>,
    edges: Vec<(u32, u32)>,
    allowed_h_milli: Vec<i32>,
}

fn topology(couplings: &[i32]) -> ScreenTopology {
    // Derived from the coordinator's Aglais preset without reordering edges.
    let fixture: TopologyFixture =
        serde_json::from_str(include_str!("fixtures/aglais.spec.json")).unwrap();
    let index: HashMap<_, _> = fixture
        .nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();
    assert_eq!(index.len(), fixture.nodes.len());
    assert_eq!(fixture.nodes.len(), 4577);
    assert_eq!(fixture.edges.len(), 41_514);
    let edges = fixture
        .edges
        .iter()
        .map(|&(u, v)| (index[&u], index[&v]))
        .collect();
    ScreenTopology::new(
        fixture.nodes.len(),
        edges,
        &fixture.allowed_h_milli,
        couplings,
    )
    .unwrap()
}

fn device() -> &'static CudaDevice {
    static DEVICE: OnceLock<CudaDevice> = OnceLock::new();
    DEVICE.get_or_init(|| {
        CudaDevice::open_with_nodes(0, KernelKind::Msa, 4577)
            .unwrap_or_else(|error| panic!("CUDA device 0: {error}"))
    })
}

fn verify(reads_log2: u32, couplings: &[i32], threads: u32) {
    let topo = topology(couplings);
    let params = ScreenParams {
        reads_log2,
        waves: 1,
        threads,
        ..ScreenParams::default()
    };
    let mut screen = Screen::open(device(), &topo, params).unwrap();
    let nonces: Vec<[u8; 32]> = (0..screen.batch())
        .map(|i| {
            let mut nonce = [0xa5; 32];
            nonce[..8].copy_from_slice(&u64::try_from(i + 1).unwrap().to_le_bytes());
            nonce
        })
        .collect();
    let run = screen.run(&nonces, 7, true).unwrap();
    let (masks, state) = run.group0_debug.expect("debug output");
    let npg = 32usize >> reads_log2;
    let want = reference_masks(&nonces[..npg], reads_log2, &topo, couplings).unwrap();
    assert_eq!(masks.len(), topo.edges().len());
    assert_eq!(state.len(), topo.nodes());
    assert_eq!(run.group0_lanes.len(), 32);
    assert_eq!(run.best.len(), nonces.len());
    for (edge, (&got, &expected)) in masks.iter().zip(&want).enumerate() {
        assert_eq!(got, expected, "coupling mask at edge {edge}");
    }
    for (lane, &energy) in run.group0_lanes.iter().enumerate() {
        let expected = reference_energy(nonces[lane % npg], &state, lane, &topo, couplings)
            .expect("protocol energy");
        assert_eq!(i64::from(energy), expected, "energy at lane {lane}");
    }
    for nonce in 0..npg {
        let best = run
            .group0_lanes
            .iter()
            .skip(nonce)
            .step_by(npg)
            .copied()
            .min()
            .unwrap();
        assert_eq!(run.best[nonce], best, "minimum for nonce {nonce}");
    }
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

//! Portfolio ground energies from SCIP, checked through both MSA storage paths.

use quip_solver_cuda::capacity::{msa_device_reads, MsaStorage};
use quip_solver_cuda::cuda_device::CudaDevice;
use quip_solver_cuda::model_file::read_model;
use quip_solver_cuda::streaming::{last_storage, sample_one_real, sample_one_real_forced};
use quip_solver_cuda::{KernelKind, SampleParams};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Deserialize)]
struct GroundState {
    file: String,
    n: usize,
    energy: Option<f64>,
    gap: Option<f64>,
    status: String,
    seconds: f64,
}

impl GroundState {
    fn proven(&self) -> bool {
        self.status == "optimal" && self.gap == Some(0.0)
    }
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/portfolio")
}

fn ground_states() -> Vec<GroundState> {
    let bytes = std::fs::read(fixtures().join("ground_states.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn portfolio_fixture_manifest_is_complete() {
    let states = ground_states();
    assert_eq!(
        states.len(),
        108,
        "expected 24 Family B and 84 Family B2 draws"
    );
    assert_eq!(
        states
            .iter()
            .filter(|s| s.file.starts_with("family-b-"))
            .count(),
        24
    );
    assert_eq!(
        states
            .iter()
            .filter(|s| s.file.starts_with("family-b2-"))
            .count(),
        84
    );
    let mut names = BTreeSet::new();
    let mut proven = 0;
    for state in states {
        assert_eq!(std::path::Path::new(&state.file).components().count(), 1);
        assert!(names.insert(state.file.clone()), "duplicate {}", state.file);
        let model = read_model(&fixtures().join(&state.file)).unwrap();
        assert_eq!(model.graph.num_nodes(), state.n, "{}", state.file);
        assert!(state.n > 0 && state.n <= 56, "{}", state.file);
        assert!(model.graph.j.len() <= state.n * (state.n - 1) / 2);
        let edges: BTreeSet<_> = model.graph.edges.iter().copied().collect();
        assert_eq!(edges.len(), model.graph.edges.len());
        assert!(state.seconds.is_finite() && state.seconds >= 0.0);
        if state.proven() {
            assert!(state.energy.unwrap().is_finite(), "{}", state.file);
            proven += 1;
        }
    }
    assert!(proven > 0, "no proven ground states");
    let files: BTreeSet<_> = std::fs::read_dir(fixtures())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".qmodel"))
        .collect();
    assert_eq!(files, names);
}

fn verify_portfolio(forced: bool) {
    let states = ground_states();
    assert_eq!(states.len(), 108);
    let device = CudaDevice::open_with_nodes(0, KernelKind::Msa, 56).unwrap();
    let params = SampleParams {
        num_reads: msa_device_reads(&device.limits),
        num_sweeps: 8192,
        sweeps_per_beta: 128,
        seed: 42,
        ..SampleParams::default()
    };
    let mut verified = 0;
    let mut failures = Vec::new();
    for state in states.iter().filter(|state| state.proven()) {
        let model = read_model(&fixtures().join(&state.file)).unwrap();
        let reads = if forced {
            sample_one_real_forced(&device, &model.graph, &params, MsaStorage::Dense)
        } else {
            sample_one_real(&device, &model.graph, &params)
        }
        .unwrap_or_else(|error| panic!("{}: {error}", state.file));
        assert_eq!(reads.len(), params.num_reads, "{}", state.file);
        assert_eq!(
            last_storage(&device),
            Some(if forced {
                MsaStorage::Dense
            } else {
                MsaStorage::Sparse
            }),
            "{}",
            state.file
        );
        let best = reads
            .iter()
            .map(|read| read.energy)
            .fold(f64::INFINITY, f64::min)
            + model.offset;
        let expected = state.energy.unwrap();
        let tolerance = 1e-6 * expected.abs().max(1.0);
        eprintln!("{}: best={best:.17} proven={expected:.17}", state.file);
        if (best - expected).abs() > tolerance {
            failures.push(format!(
                "{}: best={best:.17}, proven={expected:.17}, tolerance={tolerance}",
                state.file
            ));
        }
        verified += 1;
    }
    assert!(verified > 0, "no proven models exercised");
    assert!(
        failures.is_empty(),
        "{} of {verified} models failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn portfolio_dense_reaches_qubo_ground_state() {
    verify_portfolio(false);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn portfolio_dense_path_forced() {
    verify_portfolio(true);
}

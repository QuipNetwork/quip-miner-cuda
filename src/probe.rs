//! MSA capacity grid derived from the session allocation model.

use crate::capacity::{
    msa_dynamic_shared_bytes, msa_footprint, msa_max_model, msa_spin_layout, Connectivity,
    DeviceLimits, MsaStorage, Precision, MSA_DEFAULT_BETAS,
};

/// One capacity cell at fixed connectivity, precision, reads, and beta count.
#[derive(Debug, Clone)]
pub struct ProbeCell {
    /// Topology family.
    pub connectivity: Connectivity,
    /// Coefficient storage and arithmetic tier.
    pub precision: Precision,
    /// Sparse CSR or dense matrix.
    pub storage: MsaStorage,
    /// Whether sparse spins extend beyond shared memory.
    pub spill: bool,
    /// Largest supported variable count.
    pub nodes: usize,
    /// Couplers at that variable count.
    pub couplers: usize,
    /// Global bytes allocated per instance.
    pub per_instance_bytes: usize,
    /// RTX A4000 slowdown per lane-update at 128 reads and 48 nonces.
    /// Relative to degree-20 Binary N = 5791, within the measured range.
    pub expected_slowdown: Option<f64>,
}

impl ProbeCell {
    /// Display tier for the model's spin and coefficient storage.
    #[must_use]
    pub fn tier(&self) -> &'static str {
        match (self.storage, self.spill) {
            (MsaStorage::Dense, _) => "dense",
            (MsaStorage::Sparse, true) => "spill",
            (MsaStorage::Sparse, false) => "shared",
        }
    }
}

/// Degree 20, degree 60, and complete models by Binary, Int8, Int32, and Float32.
#[must_use]
pub fn probe(limits: &DeviceLimits, reads: usize) -> Vec<ProbeCell> {
    let mut cells = Vec::with_capacity(12);
    for connectivity in [
        Connectivity::Degree(20),
        Connectivity::Degree(60),
        Connectivity::Complete,
    ] {
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let shape = msa_max_model(connectivity, precision, reads, limits);
            let storage = connectivity.storage();
            let footprint =
                msa_footprint(shape, precision, storage, reads, MSA_DEFAULT_BETAS, limits);
            let (_, shared_spins) = msa_spin_layout(
                shape.nodes,
                reads,
                msa_dynamic_shared_bytes(limits.shared_bytes_per_block_optin),
                None,
            );
            cells.push(ProbeCell {
                connectivity,
                precision,
                storage,
                spill: storage == MsaStorage::Sparse && shared_spins < shape.nodes,
                nodes: shape.nodes,
                couplers: shape.couplers,
                per_instance_bytes: footprint.per_instance_bytes,
                expected_slowdown: measured_slowdown(connectivity, precision, shape.nodes, reads),
            });
        }
    }
    cells
}

fn measured_slowdown(
    connectivity: Connectivity,
    precision: Precision,
    nodes: usize,
    reads: usize,
) -> Option<f64> {
    if reads != 128 {
        return None;
    }
    interpolate(
        crate::slowdown_a4000::samples(connectivity, precision),
        nodes,
    )
}

#[allow(clippy::cast_precision_loss)] // supported node counts fit exactly in f64
fn interpolate(samples: &[(usize, f64)], nodes: usize) -> Option<f64> {
    let &(first, _) = samples.first()?;
    let &(last, _) = samples.last()?;
    if nodes < first || nodes > last {
        return None;
    }
    for &(n, slowdown) in samples {
        if nodes == n {
            return Some(slowdown);
        }
    }
    for pair in samples.windows(2) {
        let (lo, slow_lo) = pair[0];
        let (hi, slow_hi) = pair[1];
        if nodes > lo && nodes < hi {
            let fraction = (nodes as f64 / lo as f64).ln() / (hi as f64 / lo as f64).ln();
            return Some(slow_lo + fraction * (slow_hi - slow_lo));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::{DeviceLimits, Precision};

    fn a4000_limits() -> DeviceLimits {
        DeviceLimits {
            shared_bytes_per_block: 49_152,
            shared_bytes_per_block_optin: 101_376,
            free_bytes: 15 << 30,
            sm_count: 48,
            threads_per_sm: 1536,
        }
    }

    #[test]
    fn probe_grid_has_twelve_cells_and_lower_degree_gives_more_nodes() {
        let limits = a4000_limits();
        let cells = probe(&limits, 128);
        assert_eq!(cells.len(), 12);
        for precision in [
            Precision::Binary,
            Precision::Int8,
            Precision::Int32,
            Precision::Float32,
        ] {
            let row: Vec<_> = cells.iter().filter(|c| c.precision == precision).collect();
            assert!(row[0].nodes > row[1].nodes && row[1].nodes > row[2].nodes);
            assert!(row[0].couplers < row[1].couplers);
        }
        for row in cells.chunks(4) {
            assert!(
                row[0].nodes >= row[1].nodes
                    && row[1].nodes > row[2].nodes
                    && row[2].nodes <= row[3].nodes
            );
        }
    }

    #[test]
    fn probe_int32_boundary_includes_integer_acceptance_tables() {
        let mut limits = a4000_limits();
        let conn = Connectivity::Degree(20);
        let shape = conn.shape(10_000);
        let float = msa_footprint(shape, Precision::Float32, conn.storage(), 128, 64, &limits);
        let integer = msa_footprint(shape, Precision::Int32, conn.storage(), 128, 64, &limits);
        assert_eq!(integer.total() - float.total(), 525_056);
        limits.free_bytes = (float.total() + 100).div_ceil(4) * 5;
        assert!(float.total() <= limits.usable_bytes());
        assert!(integer.total() > limits.usable_bytes());
        let cells = probe(&limits, 128);
        let cell = |precision| {
            cells
                .iter()
                .find(|c| c.connectivity == conn && c.precision == precision)
                .unwrap()
        };
        assert_eq!(cell(Precision::Float32).nodes, shape.nodes);
        assert!(cell(Precision::Int32).nodes < shape.nodes);
        assert_eq!(
            cell(Precision::Int32).nodes,
            msa_max_model(conn, Precision::Int32, 128, &limits).nodes
        );
    }

    #[test]
    fn slowdown_baseline_is_one() {
        assert_eq!(
            measured_slowdown(Connectivity::Degree(20), Precision::Binary, 5791, 128),
            Some(1.0)
        );
    }

    #[test]
    fn slowdown_outside_range_is_none() {
        for nodes in [0, 5790, usize::MAX] {
            assert_eq!(
                measured_slowdown(Connectivity::Degree(20), Precision::Binary, nodes, 128),
                None
            );
        }
        assert_eq!(
            measured_slowdown(Connectivity::Degree(21), Precision::Binary, 5791, 128),
            None
        );
        assert_eq!(
            measured_slowdown(Connectivity::Degree(20), Precision::Binary, 5791, 64),
            None
        );
    }

    #[test]
    fn slowdown_interpolates_in_log_nodes_and_keeps_endpoints() {
        let samples = [(100, 2.0), (10_000, 6.0)];
        assert_eq!(interpolate(&samples, 100), Some(2.0));
        assert_eq!(interpolate(&samples, 10_000), Some(6.0));
        assert!((interpolate(&samples, 1000).unwrap() - 4.0).abs() < 1e-12);
        assert_eq!(interpolate(&samples, 99), None);
        assert_eq!(interpolate(&samples, 10_001), None);
        assert_eq!(interpolate(&[], 100), None);
    }

    #[test]
    fn probe_attaches_calibration_at_measured_reads() {
        let mut limits = a4000_limits();
        limits.free_bytes = 1 << 30;
        let cells = probe(&limits, 128);
        assert_eq!(
            cells
                .iter()
                .filter(|cell| cell.expected_slowdown.is_some())
                .count(),
            9
        );
        assert!(probe(&limits, 64)
            .iter()
            .all(|cell| cell.expected_slowdown.is_none()));
    }
}

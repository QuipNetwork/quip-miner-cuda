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
    /// Measured slowdown, unavailable until calibration.
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

/// Degree 20, degree 60, and complete models by Binary, Int8, and Float32.
#[must_use]
pub fn probe(limits: &DeviceLimits, reads: usize) -> Vec<ProbeCell> {
    let mut cells = Vec::with_capacity(9);
    for connectivity in [
        Connectivity::Degree(20),
        Connectivity::Degree(60),
        Connectivity::Complete,
    ] {
        for precision in [Precision::Binary, Precision::Int8, Precision::Float32] {
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
                expected_slowdown: None,
            });
        }
    }
    cells
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
    fn probe_grid_has_nine_cells_and_lower_degree_gives_more_nodes() {
        let limits = a4000_limits();
        let cells = probe(&limits, 128);
        assert_eq!(cells.len(), 9);
        for precision in [Precision::Binary, Precision::Int8, Precision::Float32] {
            let row: Vec<_> = cells.iter().filter(|c| c.precision == precision).collect();
            assert!(row[0].nodes > row[1].nodes && row[1].nodes > row[2].nodes);
            assert!(row[0].couplers < row[1].couplers);
        }
        for row in cells.chunks(3) {
            assert!(row[0].nodes >= row[1].nodes && row[1].nodes > row[2].nodes);
            assert!(row.iter().all(|c| c.expected_slowdown.is_none()));
        }
    }
}

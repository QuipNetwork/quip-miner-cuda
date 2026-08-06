//! Node-capacity policy for the self-feeding kernels.
//!
//! The kernels size their state arrays from a `QUIP_MAX_NODES` macro that
//! `cuda_device` supplies at NVRTC compile time. This module owns every
//! bound on that value so the binaries, the device, and the backend
//! identities cannot disagree.

use quip_miner_core::Algorithm;
use thiserror::Error;

/// Shipped `unpacked_state` size in `kernels/sa.cu`. Also the floor: a
/// smaller array saves nothing measurable and only narrows what runs.
pub const SA_DEFAULT_NODES: usize = 5000;

/// Hard ceiling for SA. Measured on an A4000: 8192 runs, 16384 fails with
/// `CUDA_ERROR_ILLEGAL_ADDRESS`. The array size alone does not explain a
/// bound there, so the SA path holds a second limit that is not yet found.
/// Until it is, a larger request must fail rather than corrupt memory.
pub const SA_MAX_NODES: usize = 8192;

/// Shipped `shared_state` size in `kernels/gibbs.cu`, and the floor.
pub const GIBBS_DEFAULT_NODES: usize = 4800;

/// Bytes of shared memory the Gibbs kernel uses outside `shared_state`:
/// `s_chunk` and `s_arrival`, one `int` each.
pub const GIBBS_FIXED_SHARED_BYTES: usize = 8;

/// Why a capacity request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CapacityError {
    /// Above the algorithm's own ceiling.
    #[error("requested {requested} nodes exceeds the algorithm limit of {limit}")]
    AboveAlgorithmBound {
        /// Nodes asked for.
        requested: usize,
        /// Ceiling that applies.
        limit: usize,
    },
    /// Above what this device's shared memory can hold.
    #[error("requested {requested} nodes exceeds the device shared-memory budget of {budget}")]
    AboveDeviceBudget {
        /// Nodes asked for.
        requested: usize,
        /// Budget derived from the device.
        budget: usize,
    },
}

/// Largest Gibbs `shared_state` this device can hold, in nodes. One node is
/// one `signed char`, so bytes and nodes are the same number.
#[must_use]
pub fn gibbs_budget(shared_bytes_per_block: usize) -> usize {
    shared_bytes_per_block.saturating_sub(GIBBS_FIXED_SHARED_BYTES)
}

/// Capacity used when `--max-nodes` is absent.
#[must_use]
pub fn default_nodes(algorithm: Algorithm) -> usize {
    match algorithm {
        Algorithm::Sa => SA_DEFAULT_NODES,
        Algorithm::Gibbs => GIBBS_DEFAULT_NODES,
    }
}

/// Resolve a requested capacity against the algorithm and the device.
///
/// A request below the default is raised to the default. A request above a
/// bound is an error, never a silent clamp.
///
/// # Errors
///
/// [`CapacityError::AboveAlgorithmBound`] when SA is asked for more than
/// [`SA_MAX_NODES`]. [`CapacityError::AboveDeviceBudget`] when Gibbs is
/// asked for more than [`gibbs_budget`] allows.
pub fn resolve(
    algorithm: Algorithm,
    requested: usize,
    shared_bytes_per_block: usize,
) -> Result<usize, CapacityError> {
    let floor = default_nodes(algorithm);
    let want = requested.max(floor);
    match algorithm {
        Algorithm::Sa => {
            if want > SA_MAX_NODES {
                return Err(CapacityError::AboveAlgorithmBound {
                    requested: want,
                    limit: SA_MAX_NODES,
                });
            }
            Ok(want)
        }
        Algorithm::Gibbs => {
            let budget = gibbs_budget(shared_bytes_per_block);
            if want > budget {
                return Err(CapacityError::AboveDeviceBudget {
                    requested: want,
                    budget,
                });
            }
            Ok(want)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_miner_core::Algorithm;

    /// A4000 reports 49152 bytes of shared memory per block. `s_chunk` and
    /// `s_arrival` take 8 of them, so 49144 nodes is the Gibbs ceiling.
    /// Measured: 48000 runs, 65536 fails to compile.
    #[test]
    fn gibbs_budget_subtracts_the_fixed_shared_members() {
        assert_eq!(gibbs_budget(49152), 49144);
    }

    #[test]
    fn resolve_accepts_a_request_inside_both_bounds() {
        assert_eq!(resolve(Algorithm::Gibbs, 5640, 49152), Ok(5640));
        assert_eq!(resolve(Algorithm::Sa, 5640, 49152), Ok(5640));
    }

    /// SA has an unfixed out-of-bounds defect above 8192, so a larger
    /// request must fail rather than clamp into the broken range.
    #[test]
    fn resolve_rejects_sa_above_its_bound() {
        assert_eq!(
            resolve(Algorithm::Sa, 16384, 49152),
            Err(CapacityError::AboveAlgorithmBound {
                requested: 16384,
                limit: SA_MAX_NODES,
            })
        );
    }

    #[test]
    fn resolve_rejects_gibbs_above_the_device_budget() {
        assert_eq!(
            resolve(Algorithm::Gibbs, 65536, 49152),
            Err(CapacityError::AboveDeviceBudget {
                requested: 65536,
                budget: 49144,
            })
        );
    }

    /// A request below the default would shrink the kernel array for no
    /// gain, so the default is a floor.
    #[test]
    fn resolve_raises_a_small_request_to_the_default() {
        assert_eq!(resolve(Algorithm::Sa, 64, 49152), Ok(SA_DEFAULT_NODES));
        assert_eq!(
            resolve(Algorithm::Gibbs, 64, 49152),
            Ok(GIBBS_DEFAULT_NODES)
        );
    }

    #[test]
    fn defaults_match_the_shipped_kernel_arrays() {
        assert_eq!(default_nodes(Algorithm::Sa), 5000);
        assert_eq!(default_nodes(Algorithm::Gibbs), 4800);
    }
}

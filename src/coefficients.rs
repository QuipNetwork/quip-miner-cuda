//! Which kernel variant solves a job with its exact coefficients, and at what
//! integer scale, or why none does.
//!
//! The kernels hold integers. A job is exact on a path when every `h` and `J`,
//! in milli, is a multiple of one scale and the scaled values fit the path's
//! types. The scale is the GCD of the nonzero values, so a ±1 mining job is
//! scale 1000 and uploads the same bytes as before this module existed. The
//! beta schedule is multiplied by `scale_milli / 1000`, which keeps
//! `beta * delta` in the units the schedule was built for.
//!
//! A job no path holds is refused, never quantized: the caller turns the
//! `Err` into a Capacity reject, so the coordinator can route it to a CPU
//! solver.

use crate::KernelKind;
use quip_solver_core::IsingGraph;

/// Largest |delta| the int8 SA workspace holds: `2 * field <= 127`.
pub const INT8_MAX_DELTA: i64 = 127;

/// Largest |coefficient| an int8 slot holds.
const INT8_MAX_COEF: i64 = 127;

/// Largest local field, in scale units, the wide SA path holds. The neighbour
/// update in `kernels/sa.cu` adds `4 * J`, and `|J| <= field`, so
/// `4 * field <= i32::MAX` keeps every intermediate in `int`.
pub const WIDE_MAX_FIELD: i64 = 536_870_911;

/// Relative bound on the rounding error of `milli / 1000 * 1000`, scaled by
/// the rounded milli magnitude with a floor of one.
const MILLI_TOLERANCE: f64 = 4.0 * f64::EPSILON;

/// Which compiled kernel runs the job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoefficientPath {
    /// Today's kernels: `signed char` coefficients (and an int8 SA delta).
    Int8,
    /// `kernels/sa.cu` built with `-DQUIP_SA_WIDE`: i32 coefficients and delta.
    Wide,
}

/// The path and integer scale one job runs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoefficientPlan {
    /// The kernel variant.
    pub path: CoefficientPath,
    /// Milli per kernel integer unit. Every coefficient is a multiple of it.
    pub scale_milli: u64,
}

impl CoefficientPlan {
    /// Factor applied to every beta, so the kernel's integer delta (in units
    /// of `scale_milli` milli) sees the Boltzmann weights of the model in
    /// units. Exactly `1.0` at scale 1000.
    #[must_use]
    pub fn beta_factor(self) -> f64 {
        #[allow(clippy::cast_precision_loss)] // scale_milli <= i32::MAX, exact in f64
        let scale = self.scale_milli as f64;
        scale / 1000.0
    }
}

/// A coefficient in units as integer milli, or `None` when it is non-finite,
/// off the milli grid, or outside the `i32` wire range.
#[must_use]
pub fn to_milli(v: f64) -> Option<i64> {
    let x = v * 1000.0;
    let m = x.round();
    if !m.is_finite()
        || (x - m).abs() > MILLI_TOLERANCE * m.abs().max(1.0)
        || m.abs() > f64::from(i32::MAX)
    {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)] // checked against i32::MAX above
    Some(m as i64)
}

fn milli_all(name: &str, values: &[f64]) -> Result<Vec<i64>, String> {
    values
        .iter()
        .map(|&v| {
            to_milli(v).ok_or_else(|| {
                format!("{name} coefficient {v} is not a whole milli value in the i32 wire range")
            })
        })
        .collect()
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `max_i(|h_i| + Σ_j |J_ij|)` in milli, each endpoint of every edge counted,
/// as `quip_solver_core::ModelLimits::check` counts it.
fn max_local_field(h: &[i64], j: &[i64], edges: &[(usize, usize)]) -> i64 {
    let mut field: Vec<i64> = h.iter().map(|v| v.abs()).collect();
    for (&(u, v), &jv) in edges.iter().zip(j) {
        for node in [u, v] {
            if let Some(f) = field.get_mut(node) {
                *f += jv.abs();
            }
        }
    }
    field.into_iter().max().unwrap_or(0)
}

/// Classify one job for `kind`.
///
/// `wide_max_nodes` is the largest node count the wide SA workspace fits on
/// this device (`CudaDevice::sa_wide_max_nodes`). Zero means no wide path.
///
/// # Errors
///
/// A message naming why no path holds the job exactly. Every refusal is
/// permanent for this job on this binary.
pub fn plan(
    kind: KernelKind,
    graph: &IsingGraph,
    wide_max_nodes: usize,
) -> Result<CoefficientPlan, String> {
    let h = milli_all("h", &graph.h)?;
    let j = milli_all("J", &graph.j)?;

    let g = h
        .iter()
        .chain(&j)
        .fold(0, |acc, v| gcd(acc, v.unsigned_abs()));
    let scale_milli = if g == 0 { 1000 } else { g };
    let s = i64::try_from(scale_milli).unwrap_or(i64::MAX);
    let max_coef = h.iter().chain(&j).map(|v| v.abs() / s).max().unwrap_or(0);
    let field = max_local_field(&h, &j, &graph.edges) / s;
    let int8 = CoefficientPlan {
        path: CoefficientPath::Int8,
        scale_milli,
    };

    match kind {
        KernelKind::Msa => {
            let unit_h = h.iter().all(|v| matches!(v, -1000 | 0 | 1000));
            let unit_j = j.iter().all(|v| v.abs() == 1000);
            if unit_h && unit_j {
                Ok(CoefficientPlan {
                    path: CoefficientPath::Int8,
                    scale_milli: 1000,
                })
            } else {
                Err("msa takes only unit models: every J must be ±1 and every h -1, 0 or +1".into())
            }
        }
        KernelKind::Gibbs if max_coef <= INT8_MAX_COEF => Ok(int8),
        KernelKind::Gibbs => Err(format!(
            "gibbs: a coefficient is {max_coef} times the model's {scale_milli} milli step, above the int8 limit of {INT8_MAX_COEF}"
        )),
        KernelKind::Sa if 2 * field <= INT8_MAX_DELTA => Ok(int8),
        KernelKind::Sa if field > WIDE_MAX_FIELD => Err(format!(
            "sa: local field {field} times the model's {scale_milli} milli step exceeds the wide path's limit of {WIDE_MAX_FIELD}"
        )),
        KernelKind::Sa if wide_max_nodes == 0 => {
            Err("sa: the model needs the wide path, which this device does not offer".into())
        }
        KernelKind::Sa if graph.h.len() > wide_max_nodes => Err(format!(
            "sa: the wide path holds {wide_max_nodes} nodes on this device, the model has {}",
            graph.h.len()
        )),
        KernelKind::Sa => Ok(CoefficientPlan { path: CoefficientPath::Wide, scale_milli }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelKind;
    use quip_solver_core::IsingGraph;

    fn graph(h: Vec<f64>, j: Vec<f64>, edges: Vec<(usize, usize)>) -> IsingGraph {
        IsingGraph::new(h, j, edges)
    }

    /// `K_n` with all J = +1 and every h = `h`.
    fn complete(n: usize, h: f64) -> IsingGraph {
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((u, v));
            }
        }
        let j = vec![1.0; edges.len()];
        graph(vec![h; n], j, edges)
    }

    const NO_WIDE: usize = 0;
    const BIG_WIDE: usize = 1_000_000;

    #[test]
    fn mining_model_is_int8_at_unit_scale() {
        let g = graph(vec![1.0, -1.0, 0.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]);
        for kind in [KernelKind::Sa, KernelKind::Gibbs, KernelKind::Msa] {
            let p = plan(kind, &g, NO_WIDE).expect("mining model plans");
            assert_eq!(
                p,
                CoefficientPlan {
                    path: CoefficientPath::Int8,
                    scale_milli: 1000
                }
            );
            assert_eq!(p.beta_factor().to_bits(), 1.0_f64.to_bits(), "{kind:?}");
        }
    }

    #[test]
    fn all_zero_model_is_int8_at_unit_scale() {
        let g = graph(vec![0.0; 3], vec![0.0, 0.0], vec![(0, 1), (1, 2)]);
        let p = plan(KernelKind::Sa, &g, NO_WIDE).unwrap();
        assert_eq!(
            p,
            CoefficientPlan {
                path: CoefficientPath::Int8,
                scale_milli: 1000
            }
        );
    }

    #[test]
    fn quarter_unit_model_scales_to_250_milli() {
        let g = graph(vec![0.25, -0.75], vec![0.5], vec![(0, 1)]);
        let p = plan(KernelKind::Sa, &g, NO_WIDE).unwrap();
        assert_eq!(
            p,
            CoefficientPlan {
                path: CoefficientPath::Int8,
                scale_milli: 250
            }
        );
        assert!((p.beta_factor() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn thousand_x_mining_model_scales_but_stays_int8() {
        let g = graph(vec![1000.0, -1000.0], vec![-1000.0], vec![(0, 1)]);
        let p = plan(KernelKind::Sa, &g, NO_WIDE).unwrap();
        assert_eq!(
            p,
            CoefficientPlan {
                path: CoefficientPath::Int8,
                scale_milli: 1_000_000
            }
        );
    }

    #[test]
    fn sa_int8_holds_k64_without_field_and_k65_goes_wide() {
        // K_64, h = 0: field 63, delta 126 <= 127.
        assert_eq!(
            plan(KernelKind::Sa, &complete(64, 0.0), NO_WIDE)
                .unwrap()
                .path,
            CoefficientPath::Int8
        );
        // K_64, h = 1: field 64, delta 128 > 127.
        assert_eq!(
            plan(KernelKind::Sa, &complete(64, 1.0), BIG_WIDE)
                .unwrap()
                .path,
            CoefficientPath::Wide
        );
        // K_65, h = 0: field 64.
        assert_eq!(
            plan(KernelKind::Sa, &complete(65, 0.0), BIG_WIDE)
                .unwrap()
                .path,
            CoefficientPath::Wide
        );
        assert_eq!(
            plan(KernelKind::Sa, &complete(200, 0.0), BIG_WIDE)
                .unwrap()
                .path,
            CoefficientPath::Wide
        );
        assert_eq!(
            plan(KernelKind::Sa, &complete(16, 1.0), NO_WIDE)
                .unwrap()
                .path,
            CoefficientPath::Int8
        );
    }

    #[test]
    fn wide_needs_a_node_budget() {
        let err = plan(KernelKind::Sa, &complete(65, 0.0), NO_WIDE).unwrap_err();
        assert!(err.contains("wide"), "{err}");
        let err = plan(KernelKind::Sa, &complete(65, 0.0), 64).unwrap_err();
        assert!(err.contains("65"), "{err}");
    }

    #[test]
    fn sa_wide_bound_is_a_quarter_of_i32() {
        // One milli step (scale 1), field exactly WIDE_MAX_FIELD: accepted.
        #[allow(clippy::cast_precision_loss)] // WIDE_MAX_FIELD < i32::MAX, exact in f64
        let fits = WIDE_MAX_FIELD as f64 / 1000.0;
        let g = graph(vec![fits - 0.001, 0.0], vec![0.001], vec![(0, 1)]);
        let p = plan(KernelKind::Sa, &g, BIG_WIDE).unwrap();
        assert_eq!(
            p,
            CoefficientPlan {
                path: CoefficientPath::Wide,
                scale_milli: 1
            }
        );
        // One milli more: refused.
        let g = graph(vec![fits, 0.0], vec![0.001], vec![(0, 1)]);
        assert!(plan(KernelKind::Sa, &g, BIG_WIDE).is_err());
    }

    #[test]
    fn gibbs_accepts_127_scale_units_and_refuses_128() {
        let ok = graph(vec![127.0, 0.0], vec![1.0], vec![(0, 1)]);
        assert_eq!(
            plan(KernelKind::Gibbs, &ok, NO_WIDE).unwrap().scale_milli,
            1000
        );
        let bad = graph(vec![128.0, 0.0], vec![1.0], vec![(0, 1)]);
        assert!(plan(KernelKind::Gibbs, &bad, NO_WIDE).is_err());
        // 128 and 2 share the factor 2: scale 2000, values 64 and 1.
        let scaled = graph(vec![128.0, 0.0], vec![2.0], vec![(0, 1)]);
        assert_eq!(
            plan(KernelKind::Gibbs, &scaled, NO_WIDE)
                .unwrap()
                .scale_milli,
            2000
        );
    }

    #[test]
    fn msa_accepts_only_unit_models() {
        let unit = graph(vec![1.0, 0.0], vec![-1.0], vec![(0, 1)]);
        assert!(plan(KernelKind::Msa, &unit, NO_WIDE).is_ok());
        for (h, j) in [
            (vec![2.0, 0.0], -1.0),
            (vec![1.0, 0.0], 2.0),
            (vec![0.5, 0.0], 1.0),
            (vec![1.0, 0.0], 0.0),
        ] {
            let g = graph(h.clone(), vec![j], vec![(0, 1)]);
            let err = plan(KernelKind::Msa, &g, NO_WIDE).unwrap_err();
            assert!(err.contains("msa"), "h {h:?} j {j}: {err}");
        }
    }

    #[test]
    fn non_finite_and_sub_milli_are_refused() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0005,
            1e-7,
            1e-10,
            1.000_000_000_5,
            -2_147_483.648,
        ] {
            let g = graph(vec![bad, 0.0], vec![1.0], vec![(0, 1)]);
            for kind in [KernelKind::Sa, KernelKind::Gibbs, KernelKind::Msa] {
                assert!(plan(kind, &g, BIG_WIDE).is_err(), "{kind:?} accepted {bad}");
            }
        }
    }

    #[test]
    fn to_milli_refuses_tiny_off_grid_values() {
        assert_eq!(to_milli(1e-10), None);
    }

    #[test]
    fn to_milli_refuses_off_grid_values_near_one() {
        assert_eq!(to_milli(1.000_000_000_5), None);
    }

    #[test]
    fn to_milli_round_trips_the_wire_range() {
        for m in [
            i64::from(i32::MAX),
            -i64::from(i32::MAX),
            1,
            -1,
            0,
            999,
            123_456_789,
        ] {
            #[allow(clippy::cast_precision_loss)] // |m| <= i32::MAX is exact in f64
            let v = m as f64 / 1000.0;
            assert_eq!(to_milli(v), Some(m));
        }
    }

    #[test]
    fn field_counts_both_endpoints_like_the_sdk() {
        // A self-loop counts twice on its node, as ModelLimits::check does.
        // 32 self-loop terms of J=2 give field 128 at scale 2000 -> 64 units, delta 128 > 127.
        let g = graph(vec![0.0], vec![2.0; 32], vec![(0, 0); 32]);
        assert_eq!(
            plan(KernelKind::Sa, &g, BIG_WIDE).unwrap().path,
            CoefficientPath::Wide
        );
        // A duplicate edge counts once per copy on each endpoint.
        let g = graph(vec![0.0, 0.0], vec![1.0; 64], vec![(0, 1); 64]);
        assert_eq!(
            plan(KernelKind::Sa, &g, BIG_WIDE).unwrap().path,
            CoefficientPath::Wide
        );
    }
}

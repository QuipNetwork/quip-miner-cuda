//! Integer Metropolis for the wide SA kernel.
//!
//! `kernels/sa.cu` built with `-DQUIP_SA_WIDE` decides each flip with integer
//! arithmetic only, so a fixed seed gives the same samples on every GPU: no
//! `__expf`, whose last bits differ between architectures. The uniform draw
//! `u` is quantised to `NEGLOG_LEN` midpoints and `-ln(u)` is precomputed here
//! in 32.32 fixed point. Each rung's beta goes to the same fixed point, with a
//! cut above which no draw can accept. See `wide_rungs` for the rule.
//!
//! The quantised draw cannot accept a move whose probability is below about
//! `0.5 / NEGLOG_LEN` (7.6e-6). The int8 kernel's own cut is
//! `22.18 / beta` (about 2.3e-10). This is the price of exactness, and the
//! tests in `tests/exact_coefficients_gpu.rs` measure its effect on quality.

/// Bits of the per-decision random index into the table.
pub const NEGLOG_BITS: u32 = 16;
/// Entries in the `-ln(u)` table.
pub const NEGLOG_LEN: usize = 1 << NEGLOG_BITS;
/// Fixed-point fraction bits of the table and of `beta_fixed`.
pub const FIXED_SHIFT: u32 = 32;

#[allow(clippy::cast_precision_loss)] // 2^32 and NEGLOG_LEN are exact in f64
fn fixed_one() -> f64 {
    (1_u64 << FIXED_SHIFT) as f64
}

/// `-ln(u_k) · 2^32` for the midpoints `u_k = (k + 0.5) / NEGLOG_LEN`.
#[must_use]
pub fn neglog_table() -> Vec<i64> {
    #[allow(clippy::cast_precision_loss)]
    let len = NEGLOG_LEN as f64;
    (0..NEGLOG_LEN)
        .map(|k| {
            #[allow(clippy::cast_precision_loss)]
            let u = (k as f64 + 0.5) / len;
            #[allow(clippy::cast_possible_truncation)] // at most 11.8 · 2^32, far below i64::MAX
            let v = (-u.ln() * fixed_one()).round() as i64;
            v
        })
        .collect()
}

/// Per-rung `(beta_fixed, delta_cut)` for the kernel betas `betas`, which are
/// already multiplied by the plan's `beta_factor`.
///
/// `beta_fixed = round(beta · 2^32)`, clamped so it stays positive in i64.
/// `delta_cut = floor(table[0] / beta_fixed)`, clamped to `i32::MAX`. With
/// those, the kernel accepts a positive `de` iff `de <= delta_cut` and
/// `de · beta_fixed < table[r]`, and the product fits i64.
#[must_use]
pub fn wide_rungs(betas: &[f32]) -> (Vec<i64>, Vec<i32>) {
    let top = neglog_table()[0];
    betas
        .iter()
        .map(|&b| {
            let scaled = (f64::from(b) * fixed_one()).round();
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let beta_fixed = if scaled <= 0.0 {
                0
            } else if scaled >= i64::MAX as f64 {
                i64::MAX
            } else {
                scaled as i64
            };
            let cut = if beta_fixed == 0 {
                i64::from(i32::MAX)
            } else {
                top / beta_fixed
            };
            (beta_fixed, i32::try_from(cut).unwrap_or(i32::MAX))
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_positive_descending_and_pinned() {
        let t = neglog_table();
        assert_eq!(t.len(), NEGLOG_LEN);
        assert!(t.windows(2).all(|w| w[0] > w[1]));
        assert!(*t.last().unwrap() > 0);
        // -ln(0.5 / 65536) * 2^32 and -ln(65535.5 / 65536) * 2^32, rounded.
        assert_eq!(t[0], 50_609_756_021);
        assert_eq!(t[NEGLOG_LEN - 1], 32_768);
    }

    #[test]
    fn product_never_overflows_below_the_cut() {
        let t = neglog_table();
        for beta in [1e-6_f32, 0.01, 1.0, 37.5, 1e4, 3e9] {
            let (bf, cut) = wide_rungs(&[beta]);
            let product = i128::from(cut[0]) * i128::from(bf[0]);
            assert!(product <= i128::from(t[0]), "beta {beta}");
            let above = i128::from(cut[0]) + 1;
            assert!(
                above * i128::from(bf[0]) > i128::from(t[0]) || cut[0] == i32::MAX,
                "beta {beta}"
            );
        }
    }

    #[test]
    fn zero_beta_accepts_everything() {
        let (bf, cut) = wide_rungs(&[0.0]);
        assert_eq!((bf[0], cut[0]), (0, i32::MAX));
    }

    #[test]
    fn acceptance_rate_matches_exp_minus_beta_delta() {
        let t = neglog_table();
        let (bf, cut) = wide_rungs(&[0.5]);
        for de in [1_i64, 2, 4, 8] {
            let accepted = if de > i64::from(cut[0]) {
                0
            } else {
                t.iter().filter(|&&l| de * bf[0] < l).count()
            };
            #[allow(clippy::cast_precision_loss)]
            let rate = accepted as f64 / NEGLOG_LEN as f64;
            #[allow(clippy::cast_precision_loss)]
            let want = (-0.5 * de as f64).exp();
            #[allow(clippy::cast_precision_loss)] // NEGLOG_LEN is 2^16, exact in f64
            let tolerance = 2.0 / NEGLOG_LEN as f64;
            assert!((rate - want).abs() < tolerance, "de {de}: {rate} vs {want}");
        }
    }

    #[test]
    fn table_bytes_are_stable() {
        // The wide path promises the same samples on every GPU, so the table must
        // not move between builds. Pin a checksum of all entries.
        let sum: i128 = neglog_table().iter().map(|&v| i128::from(v)).sum();
        assert_eq!(sum, 281_473_488_191_052);
    }
}

//! Drawdown analytics. Equity curve in arbitrary units → max drawdown
//! in basis points (1bp = 1/10_000 of peak equity). Reported in bp so
//! the dashboard / DB columns stay readable across instruments.

/// Peak-to-trough drawdown along an equity curve, expressed in basis
/// points relative to the running peak. Always >= 0. Empty/single-point
/// curves return 0.
pub fn max_drawdown_bp(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        if peak > 0.0 {
            let dd = (peak - e) / peak * 10_000.0;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    max_dd
}

/// Per-point drawdown series in basis points. Same length as input.
pub fn equity_drawdown(equity: &[f64]) -> Vec<f64> {
    let mut peak = f64::NEG_INFINITY;
    let mut out = Vec::with_capacity(equity.len());
    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = if peak > 0.0 {
            (peak - e) / peak * 10_000.0
        } else {
            0.0
        };
        out.push(dd);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_curve_has_zero_drawdown() {
        assert_eq!(max_drawdown_bp(&[100.0, 100.0, 100.0]), 0.0);
    }

    #[test]
    fn known_drawdown_matches() {
        // Peak 100 → trough 90 = 1000 bp drawdown.
        let eq = [100.0, 95.0, 90.0, 95.0];
        assert!((max_drawdown_bp(&eq) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn drawdown_series_has_correct_length() {
        let eq = [100.0, 110.0, 99.0, 101.0];
        assert_eq!(equity_drawdown(&eq).len(), eq.len());
    }
}

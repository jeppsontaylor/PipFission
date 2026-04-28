//! Meta-labeling helpers (López de Prado §3.7).
//!
//! Given a primary side prediction (e.g. from an existing classifier) and
//! a triple-barrier label, the meta-label is `1` if the trade taken by
//! the primary signal would have *cleared the cost floor*, else `0`.
//! It's the second-stage classifier's target: "was the side prediction
//! worth taking after costs?"

use crate::triple_barrier::LabelRow;

/// Compute the meta label for a side prediction `predicted_side` against
/// a `LabelRow`. `min_edge` is the absolute return threshold (after
/// costs) above which the trade counts as a hit.
///
/// Returns `1` iff `predicted_side * realized_r >= min_edge`.
/// `predicted_side` should be `-1` (short) or `+1` (long); `0` (no
/// prediction) always yields `0`.
pub fn meta_label(predicted_side: i8, label: &LabelRow, min_edge: f64) -> u8 {
    if predicted_side == 0 {
        return 0;
    }
    let pnl = predicted_side as f64 * label.realized_r;
    if pnl >= min_edge {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triple_barrier::BarrierHit;

    fn lab(realized_r: f64) -> LabelRow {
        LabelRow {
            ts_ms: 0,
            t1_ms: 1,
            side: if realized_r > 0.0 { 1 } else { -1 },
            meta_y: 1,
            realized_r,
            barrier_hit: BarrierHit::Vert,
        }
    }

    #[test]
    fn long_prediction_with_positive_return_is_one() {
        assert_eq!(meta_label(1, &lab(0.005), 0.001), 1);
    }

    #[test]
    fn long_prediction_with_negative_return_is_zero() {
        assert_eq!(meta_label(1, &lab(-0.002), 0.001), 0);
    }

    #[test]
    fn no_prediction_is_zero() {
        assert_eq!(meta_label(0, &lab(0.10), 0.0), 0);
    }
}

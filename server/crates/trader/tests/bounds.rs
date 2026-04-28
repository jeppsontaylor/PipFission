//! Pin: every default in `TraderParams` must lie within its `BOUNDS`
//! entry. Catches the drift mode where someone bumps a default but
//! forgets to widen the bounds (or vice versa). Without this guard
//! the optimiser could refuse to sample the default — and the live
//! defaults would never get re-evaluated post-bump.

use trader::params::{Bound, TraderParams, BOUNDS};

/// Look up the default value for one named field. Hand-written
/// because Rust's reflection is limited; a Phase C param mismatch
/// shows up as either a compile error here (someone removed a field)
/// or a runtime panic (someone changed a default outside its bounds).
fn default_for(name: &str, p: &TraderParams) -> f64 {
    match name {
        "long_threshold" => p.long_threshold,
        "short_threshold" => p.short_threshold,
        "take_threshold" => p.take_threshold,
        "min_conf_margin" => p.min_conf_margin,
        "stop_loss_atr" => p.stop_loss_atr,
        "take_profit_atr" => p.take_profit_atr,
        "trailing_stop_atr" => p.trailing_stop_atr,
        "min_hold_bars" => p.min_hold_bars as f64,
        "max_hold_bars" => p.max_hold_bars as f64,
        "cooldown_bars" => p.cooldown_bars as f64,
        "max_position_frac" => p.max_position_frac,
        "daily_loss_limit_bp" => p.daily_loss_limit_bp,
        "max_dd_pause_bp" => p.max_dd_pause_bp,
        "spread_max_bp" => p.spread_max_bp,
        "stale_data_ms" => p.stale_data_ms as f64,
        other => panic!("BOUNDS includes {other:?} but the test has no field accessor"),
    }
}

#[test]
fn every_default_is_inside_its_bounds() {
    let defaults = TraderParams::default();
    for b in BOUNDS {
        let v = default_for(b.name, &defaults);
        assert!(
            v >= b.lo && v <= b.hi,
            "default for {} ({}) is outside bounds [{}, {}]",
            b.name,
            v,
            b.lo,
            b.hi,
        );
    }
}

#[test]
fn min_hold_bound_makes_scalp_unrepresentable() {
    // Phase C structural guarantee: the optimiser cannot pick
    // min_hold_bars below 10 (= 100 seconds on 10s bars). Pinning
    // this here means a future relaxation requires touching this
    // test, which is the right level of friction.
    let b = BOUNDS
        .iter()
        .find(|b| b.name == "min_hold_bars")
        .expect("min_hold_bars in BOUNDS");
    assert!(b.lo >= 10.0, "min_hold_bars lower bound regressed to {}", b.lo);
}

#[test]
fn max_hold_bound_makes_short_window_unrepresentable() {
    // Companion guarantee: max_hold_bars cannot collapse below 60
    // (= 10 minutes on 10s bars). Prevents the optimiser from
    // recreating the old "scalp the next 12 bars" regime.
    let b = BOUNDS
        .iter()
        .find(|b| b.name == "max_hold_bars")
        .expect("max_hold_bars in BOUNDS");
    assert!(b.lo >= 60.0, "max_hold_bars lower bound regressed to {}", b.lo);
}

#[test]
fn min_conf_margin_floor_pinned() {
    // No 0%-confidence-margin entries.
    let b = BOUNDS
        .iter()
        .find(|b| b.name == "min_conf_margin")
        .expect("min_conf_margin in BOUNDS");
    assert!(b.lo >= 0.05, "min_conf_margin lower bound regressed to {}", b.lo);
}

#[test]
fn cooldown_floor_pinned() {
    let b = BOUNDS
        .iter()
        .find(|b| b.name == "cooldown_bars")
        .expect("cooldown_bars in BOUNDS");
    assert!(b.lo >= 6.0, "cooldown_bars lower bound regressed to {}", b.lo);
}

#[test]
fn bounds_lo_is_le_hi() {
    // Sanity: no inverted entries in the table.
    for b in BOUNDS {
        assert!(b.lo < b.hi, "bound {} has lo={} not < hi={}", b.name, b.lo, b.hi);
    }
}

#[test]
fn integer_bounds_have_integer_defaults() {
    // If a Bound declares `is_int: true`, the corresponding default
    // must already be an integer value (no fractional 12.5 sneaking in).
    let defaults = TraderParams::default();
    for b in BOUNDS {
        if !b.is_int {
            continue;
        }
        let v = default_for(b.name, &defaults);
        assert_eq!(
            v.fract(),
            0.0,
            "integer bound {} has non-integer default {}",
            b.name,
            v,
        );
    }
}

#[test]
fn min_hold_strictly_less_than_max_hold() {
    let defaults = TraderParams::default();
    assert!(
        defaults.min_hold_bars < defaults.max_hold_bars,
        "min_hold_bars {} must be < max_hold_bars {}",
        defaults.min_hold_bars,
        defaults.max_hold_bars,
    );
}

#[allow(dead_code)]
fn _bound_type_check(_: Bound) {
    // Catches the case where Bound's struct shape changes — the
    // pattern in `default_for` would no longer compile.
}

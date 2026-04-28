//! Cost model. Applied to every Open/Close fill in the backtester so the
//! `realized_r` reported by the trader is *gross*; the backtester then
//! subtracts costs before the metrics summary.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Costs {
    /// Commission in basis points per round trip (entry + exit).
    pub commission_bp: f64,
    /// Half-spread paid per fill in basis points; entry + exit each pay half-spread.
    pub spread_bp: f64,
    /// Random-walk-style slippage in basis points applied per fill.
    pub slippage_bp: f64,
}

impl Default for Costs {
    fn default() -> Self {
        Self {
            // 0.5 bp commission round-trip is generous for OANDA forex /
            // crypto on Alpaca. The optimiser also stress-tests at 2× and 3×.
            commission_bp: 0.5,
            // OANDA practice spreads are typically 0.5–2 bp on majors.
            spread_bp: 1.0,
            // Generous slippage default; optimiser stress-tests up.
            slippage_bp: 0.5,
        }
    }
}

impl Costs {
    /// Total cost of a complete round trip in basis points.
    pub fn round_trip_bp(&self) -> f64 {
        self.commission_bp + 2.0 * (self.spread_bp + self.slippage_bp)
    }

    /// Cost as a return-fraction (i.e. /10_000).
    pub fn round_trip_frac(&self) -> f64 {
        self.round_trip_bp() / 10_000.0
    }
}

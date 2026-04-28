//! 3-way reconciliation between StrategyExpected, InternalPaperActual,
//! and OandaActual. Stub in M1; populated in M6.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reconciliation {
    /// What strategy thought the book was at signal-emission time.
    pub strategy_expected_balance: f64,
    /// What internal PaperBook says after fills.
    pub internal_paper_actual_balance: f64,
    /// What OANDA's actual account NAV reports.
    pub oanda_actual_balance: f64,
    pub internal_minus_strategy_bp: f64,
    pub oanda_minus_internal_bp: f64,
}

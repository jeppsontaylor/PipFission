//! Per-instrument incremental feature state.
//!
//! Holds one instance of each indicator at the appropriate window size,
//! plus tiny lag buffers for the 1/10/60-tick log returns. `push(tick)`
//! is O(1) per instrument and returns `Some(FeatureVector)` once warmup
//! has completed (>= 300 ticks).

use std::collections::VecDeque;

use chrono::{DateTime, Datelike, Timelike, Utc};

use market_domain::{FeatureVector, OrderBookSnapshot, FEATURE_DIM};

use crate::indicators::{Ema, Macd, Sma};

const WIN_60: usize = 60;
const WIN_300: usize = 300;
const RSI_P: usize = 14;
const ATR_P: usize = 14;
const MACD_FAST: usize = 60;
const MACD_SLOW: usize = 300;
const MACD_SIG: usize = 9;
const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct IncrementalFeatures {
    instrument: String,

    // Lag buffer for 1/10/60-tick log returns. We keep the last 60 mids.
    mid_lag: VecDeque<f64>,

    sma60: Sma,
    sma300: Sma,
    macd: Macd,
    atr_ema: Ema, // ATR is EMA of |range_per_tick|

    // Cached RSI (Rsi has no `last()` accessor; we cache the value returned
    // by its push so we can read it later in the same tick).
    last_rsi: Option<f64>,
    rsi: crate::indicators::Rsi,

    // tick velocity: count of ticks seen in the last second. We approximate
    // with a sliding window of (timestamp, count=1) over 60 ticks and divide
    // by elapsed seconds.
    tick_times: VecDeque<DateTime<Utc>>,

    /// Latest order book snapshot (crypto only). When present, we compute
    /// the 6 orderbook features in slots 18..24; otherwise we leave them
    /// as zeros.
    orderbook: Option<OrderBookSnapshot>,

    /// How many ticks observed (used to determine warmup). Saturates.
    seen: u64,
}

impl IncrementalFeatures {
    pub fn new(instrument: String) -> Self {
        Self {
            instrument,
            mid_lag: VecDeque::with_capacity(WIN_60 + 1),
            sma60: Sma::new(WIN_60),
            sma300: Sma::new(WIN_300),
            macd: Macd::new(MACD_FAST, MACD_SLOW, MACD_SIG),
            atr_ema: Ema::from_period(ATR_P),
            last_rsi: None,
            rsi: crate::indicators::Rsi::new(RSI_P),
            tick_times: VecDeque::with_capacity(WIN_60 + 1),
            orderbook: None,
            seen: 0,
        }
    }

    /// Register the latest order book snapshot for this instrument.
    /// Cached and used the next time `push()` is called.
    pub fn set_orderbook(&mut self, ob: OrderBookSnapshot) {
        self.orderbook = Some(ob);
    }

    pub fn instrument(&self) -> &str {
        &self.instrument
    }

    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Push a (mid, ask, bid, time) tick. Returns `Some(FeatureVector)`
    /// once the per-instrument warmup window of 300 ticks has been reached.
    pub fn push(
        &mut self,
        mid: f64,
        bid: f64,
        ask: f64,
        time: DateTime<Utc>,
    ) -> Option<FeatureVector> {
        self.seen = self.seen.saturating_add(1);

        // Lag buffer: keep last 61 mids so we can look back 60 ticks.
        self.mid_lag.push_back(mid);
        if self.mid_lag.len() > WIN_60 + 1 {
            self.mid_lag.pop_front();
        }

        // Tick times for velocity.
        self.tick_times.push_back(time);
        if self.tick_times.len() > WIN_60 {
            self.tick_times.pop_front();
        }

        // Log returns over various lags, computed against the lag buffer.
        let log_ret = |lag: usize| -> f64 {
            if mid <= 0.0 || self.mid_lag.len() <= lag {
                return 0.0;
            }
            let prev = self.mid_lag[self.mid_lag.len() - 1 - lag];
            if prev <= 0.0 {
                0.0
            } else {
                (mid / prev).ln()
            }
        };
        let lr1 = log_ret(1);
        let lr10 = log_ret(10);
        let lr60 = log_ret(60);

        // Update rolling stats. We compute most variances/ranges directly
        // from `mid_lag` to keep things simple; only EMAs / MACD / RSI use
        // their dedicated structs.
        let _ = self.sma60.push(mid);
        let _ = self.sma300.push(mid);
        if let Some(r) = self.rsi.push(mid) {
            self.last_rsi = Some(r);
        }
        let (macd_line, macd_sig) = self.macd.push(mid);
        // ATR proxy: use bid/ask spread as a per-tick "true range".
        let _atr_now = self.atr_ema.push(ask - bid);

        // Until warmup reached, don't emit a feature vector.
        if !self.is_warmed_up() {
            return None;
        }

        let mid_safe = if mid == 0.0 { f64::EPSILON } else { mid };

        // Realized vol = sqrt(variance) of 1-tick log returns. We compute
        // from `mid_lag` directly to avoid mutating the rolling-var struct
        // a second time per tick (its push is already done above).
        let realized_vol_60 = self.var60_value().sqrt();
        let realized_vol_300 = self.var300_value().sqrt();

        let spread_now = ask - bid;
        let (mean_sp, var_sp) = self.spread_var300_value();
        let std_sp = var_sp.sqrt();
        let spread_zscore_300 = if std_sp > 1e-12 {
            (spread_now - mean_sp) / std_sp
        } else {
            0.0
        };
        let spread_bp = if mid_safe.abs() > 1e-12 {
            (spread_now / mid_safe) * 10_000.0
        } else {
            0.0
        };

        let sma60 = self.sma60.last().unwrap_or(mid);
        let sma300 = self.sma300.last().unwrap_or(mid);
        let mid_minus_sma60_over_mid = (mid - sma60) / mid_safe;
        let mid_minus_sma300_over_mid = (mid - sma300) / mid_safe;

        let rsi_v = self.rsi_last().unwrap_or(50.0);

        // Bollinger %B over 300-tick window: (mid - lower) / (upper - lower)
        // where bands = mean ± 2*std on mid.
        let (bb_mean, bb_var) = self.bb_var_value();
        let bb_std = bb_var.sqrt();
        let lower = bb_mean - 2.0 * bb_std;
        let upper = bb_mean + 2.0 * bb_std;
        let bollinger_pct_b = if upper > lower {
            (mid - lower) / (upper - lower)
        } else {
            0.5
        };

        let atr14 = self.atr_ema.last().unwrap_or(spread_now);

        let macd_signal_minus_line = macd_sig - macd_line;

        // Tick velocity: ticks per second over the window.
        let tick_velocity = if self.tick_times.len() >= 2 {
            let span_s = (*self.tick_times.back().unwrap() - *self.tick_times.front().unwrap())
                .num_milliseconds() as f64
                / 1000.0;
            if span_s > 1e-9 {
                self.tick_times.len() as f64 / span_s
            } else {
                0.0
            }
        } else {
            0.0
        };

        // 60-tick range from mid_lag directly (avoid double-push side effect).
        let (rng_lo, rng_hi) = if self.mid_lag.len() >= WIN_60 {
            let start = self.mid_lag.len() - WIN_60;
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for i in start..self.mid_lag.len() {
                let v = self.mid_lag[i];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            (lo, hi)
        } else {
            (mid, mid)
        };
        let range60_over_mid = (rng_hi - rng_lo) / mid_safe;

        // Intraday seasonality: minute-of-hour sin/cos.
        let minute = time.minute() as f64 + (time.second() as f64 / 60.0);
        let theta = 2.0 * std::f64::consts::PI * minute / 60.0;
        let mh_sin = theta.sin();
        let mh_cos = theta.cos();

        let _ = time.month(); // silence unused-method warning if any

        // Orderbook features (zeros for forex pairs without depth).
        let (ob_imb, ob_dwm_bp, ob_bid_wall, ob_ask_wall, ob_cum10, ob_cum50) = self
            .orderbook
            .as_ref()
            .map(|ob| compute_ob_features(ob, mid))
            .unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

        let vec: [f64; FEATURE_DIM] = [
            lr1,
            lr10,
            lr60,
            realized_vol_60,
            realized_vol_300,
            spread_bp,
            spread_zscore_300,
            mid_minus_sma60_over_mid,
            mid_minus_sma300_over_mid,
            rsi_v,
            bollinger_pct_b,
            atr14,
            macd_line,
            macd_signal_minus_line,
            tick_velocity,
            range60_over_mid,
            mh_sin,
            mh_cos,
            // 18..24 orderbook features
            ob_imb,
            ob_dwm_bp,
            ob_bid_wall,
            ob_ask_wall,
            ob_cum10,
            ob_cum50,
        ];

        Some(FeatureVector {
            instrument: self.instrument.clone(),
            time,
            version: SCHEMA_VERSION,
            vector: vec,
        })
    }

    pub fn is_warmed_up(&self) -> bool {
        self.seen as usize >= crate::WARMUP_TICKS
    }

    // The double-push pattern in `push` above is awkward; below are read-only
    // accessors that recompute (cheap) instead of mutating. They're stand-ins
    // until the indicators expose `peek()`.
    fn var60_value(&self) -> f64 {
        // We can't read RollingVar without mutating; instead, recompute from
        // mid_lag (last 60 log returns). O(60) — fine per tick.
        if self.mid_lag.len() <= WIN_60 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 1..=WIN_60 {
            let here = self.mid_lag[self.mid_lag.len() - i];
            let prev = self.mid_lag[self.mid_lag.len() - i - 1];
            let lr = if prev > 0.0 { (here / prev).ln() } else { 0.0 };
            sum += lr;
            sum_sq += lr * lr;
        }
        let n = WIN_60 as f64;
        let mean = sum / n;
        (sum_sq / n - mean * mean).max(0.0)
    }
    fn var300_value(&self) -> f64 {
        // For 300-tick variance we'd need 300 mids; mid_lag only has 60.
        // Approximate by scaling the 60-tick variance by sqrt(300/60). Good
        // enough for a feature; documented as approximation.
        self.var60_value() * 5.0
    }
    fn spread_var300_value(&self) -> (f64, f64) {
        // Approximate: assume spread ≈ stable; report current spread as
        // mean and 0 variance. Real 300-tick rolling spread variance would
        // need an extra ring. Sprint 3 will replace.
        (0.0, 0.0)
    }
    fn rsi_last(&self) -> Option<f64> {
        self.last_rsi.or(Some(50.0))
    }
    fn bb_var_value(&self) -> (f64, f64) {
        // Approximate: use SMA300 as mean and var60 * 5 as variance (rough).
        let mean = self.sma300.last().unwrap_or(0.0);
        (mean, self.var60_value() * 5.0)
    }
}

/// Returns (imbalance, depth_weighted_mid_bp, bid_wall, ask_wall, cum10bp, cum50bp).
fn compute_ob_features(ob: &OrderBookSnapshot, mid: f64) -> (f64, f64, f64, f64, f64, f64) {
    let safe_mid = if mid > 0.0 { mid } else { 1.0 };
    let bids = &ob.bids.levels;
    let asks = &ob.asks.levels;
    if bids.is_empty() || asks.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    // Top-of-book imbalance.
    let (_, top_bid_size) = bids[0];
    let (_, top_ask_size) = asks[0];
    let imb_denom = top_bid_size + top_ask_size;
    let imb = if imb_denom > 0.0 {
        (top_bid_size - top_ask_size) / imb_denom
    } else {
        0.0
    };

    // Depth-weighted mid: each side weighted by total size up to N levels.
    let take_levels = 10usize;
    let (bid_pw, bid_w) = bids
        .iter()
        .take(take_levels)
        .fold((0.0_f64, 0.0_f64), |(pw, w), (p, s)| (pw + p * s, w + s));
    let (ask_pw, ask_w) = asks
        .iter()
        .take(take_levels)
        .fold((0.0_f64, 0.0_f64), |(pw, w), (p, s)| (pw + p * s, w + s));
    let dwm = if (bid_w + ask_w) > 0.0 {
        (bid_pw + ask_pw) / (bid_w + ask_w)
    } else {
        safe_mid
    };
    let dwm_bp = (dwm - safe_mid) / safe_mid * 10_000.0;

    // Wall ratios: max(size) / mean(size) over top 20.
    fn wall(levels: &[(f64, f64)]) -> f64 {
        if levels.is_empty() {
            return 0.0;
        }
        let mut max = 0.0_f64;
        let mut sum = 0.0_f64;
        for (_, s) in levels.iter().take(20) {
            if *s > max {
                max = *s;
            }
            sum += s;
        }
        let n = levels.len().min(20) as f64;
        if sum > 0.0 {
            max / (sum / n)
        } else {
            0.0
        }
    }
    let bid_wall = wall(bids);
    let ask_wall = wall(asks);

    // Cumulative depth within ±X bp of mid.
    fn cum_depth(levels: &[(f64, f64)], mid: f64, bp: f64) -> f64 {
        let bound = mid * (bp / 10_000.0);
        let lo = mid - bound;
        let hi = mid + bound;
        levels
            .iter()
            .filter(|(p, _)| *p >= lo && *p <= hi)
            .map(|(_, s)| s)
            .sum()
    }
    let cum10 = cum_depth(bids, safe_mid, 10.0) + cum_depth(asks, safe_mid, 10.0);
    let cum50 = cum_depth(bids, safe_mid, 50.0) + cum_depth(asks, safe_mid, 50.0);

    (imb, dwm_bp, bid_wall, ask_wall, cum10, cum50)
}

#[cfg(test)]
mod ob_tests {
    use super::*;
    use chrono::Utc;
    use market_domain::OrderBookSide;

    #[test]
    fn imbalance_balanced_orderbook_is_zero() {
        let ob = OrderBookSnapshot {
            instrument: "BTC/USD".into(),
            time: Utc::now(),
            bids: OrderBookSide {
                levels: vec![(50000.0, 1.0), (49999.0, 1.0)],
            },
            asks: OrderBookSide {
                levels: vec![(50001.0, 1.0), (50002.0, 1.0)],
            },
        };
        let (imb, _, _, _, _, _) = compute_ob_features(&ob, 50000.5);
        assert!(imb.abs() < 1e-9);
    }

    #[test]
    fn imbalance_bid_heavy_is_positive() {
        let ob = OrderBookSnapshot {
            instrument: "BTC/USD".into(),
            time: Utc::now(),
            bids: OrderBookSide {
                levels: vec![(50000.0, 5.0)],
            },
            asks: OrderBookSide {
                levels: vec![(50001.0, 1.0)],
            },
        };
        let (imb, _, _, _, _, _) = compute_ob_features(&ob, 50000.5);
        assert!(imb > 0.5);
    }

    #[test]
    fn cum_depth_filters_by_band() {
        let ob = OrderBookSnapshot {
            instrument: "BTC/USD".into(),
            time: Utc::now(),
            bids: OrderBookSide {
                levels: vec![(50000.0, 1.0), (49500.0, 100.0)], // 49500 is 100bp away from 50000.5
            },
            asks: OrderBookSide {
                levels: vec![(50001.0, 1.0), (50500.0, 100.0)],
            },
        };
        let (_, _, _, _, cum10, _cum50) = compute_ob_features(&ob, 50000.5);
        // ±10bp of 50000.5 = ±50.0005 → bounds [49950.5, 50050.5] → only top
        // levels (50000, 50001) are inside — 1.0 + 1.0 = 2.0
        assert!((cum10 - 2.0).abs() < 1e-6, "cum10={cum10}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(s: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(s, 0).unwrap()
    }

    #[test]
    fn no_features_before_warmup() {
        let mut f = IncrementalFeatures::new("EUR_USD".into());
        for i in 0..(crate::WARMUP_TICKS - 1) {
            let mid = 1.10 + (i as f64) * 1e-5;
            assert!(f.push(mid, mid - 1e-4, mid + 1e-4, ts(i as i64)).is_none());
        }
    }

    #[test]
    fn features_after_warmup() {
        let mut f = IncrementalFeatures::new("EUR_USD".into());
        let mut last = None;
        for i in 0..(crate::WARMUP_TICKS + 50) {
            let mid = 1.10 + (i as f64).sin() * 1e-3;
            last = f.push(mid, mid - 1e-4, mid + 1e-4, ts(i as i64));
        }
        let v = last.expect("should emit features after warmup");
        assert_eq!(v.instrument, "EUR_USD");
        assert_eq!(v.vector.len(), FEATURE_DIM);
        // None of the features should be NaN.
        for (i, x) in v.vector.iter().enumerate() {
            assert!(
                x.is_finite(),
                "feature {i} ({}) is non-finite: {x}",
                market_domain::FEATURE_NAMES[i]
            );
        }
    }
}

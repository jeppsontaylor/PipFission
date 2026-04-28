//! Maps incoming `StrategySignal`s into `OrderIntent`s subject to a
//! confidence threshold, a per-instrument cooldown, and an exposure cap.
//! Lives in `strategy` so it shares the model context. The intent itself
//! is a pure value type defined in `market-domain::order`.
//!
//! Sprint 2 milestone M5. The intent stream is consumed by
//! `api-server::commands::intent_runner` which actually invokes the router.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use market_domain::{SignalDirection, StrategySignal};

#[derive(Clone, Copy, Debug)]
pub struct EmitterConfig {
    pub confidence_threshold: f64,
    pub cooldown_secs: i64,
    pub units_per_signal: i64,
    /// Max net |units| held at any time per instrument.
    pub max_position: i64,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.55,
            cooldown_secs: 30,
            units_per_signal: 100,
            max_position: 1000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmittedIntent {
    pub instrument: String,
    /// Signed units delta to apply (positive = buy, negative = sell, 0 = no-op).
    pub units_delta: i64,
    pub triggered_by_model: String,
}

#[derive(Default)]
pub struct IntentEmitter {
    cfg: EmitterConfig,
    /// instrument -> (last emitted direction, time, current net position)
    state: HashMap<String, EmitterState>,
}

#[derive(Clone, Debug)]
struct EmitterState {
    last_direction: SignalDirection,
    last_time: Option<DateTime<Utc>>,
    net_units: i64,
}

impl IntentEmitter {
    pub fn new() -> Self {
        Self {
            cfg: EmitterConfig::default(),
            state: HashMap::new(),
        }
    }

    pub fn with_config(cfg: EmitterConfig) -> Self {
        Self {
            cfg,
            state: HashMap::new(),
        }
    }

    /// Set the *known* current position for an instrument (called when a
    /// fill happens so the emitter respects the position cap).
    pub fn set_position(&mut self, instrument: &str, net_units: i64) {
        self.state
            .entry(instrument.to_string())
            .or_insert_with(|| EmitterState {
                last_direction: SignalDirection::Flat,
                last_time: None,
                net_units,
            })
            .net_units = net_units;
    }

    /// Process a signal. Returns `Some(EmittedIntent)` if the signal should
    /// trigger a new order; `None` otherwise (cooldown, threshold, or
    /// already in target direction).
    pub fn on_signal(&mut self, sig: &StrategySignal) -> Option<EmittedIntent> {
        if sig.confidence < self.cfg.confidence_threshold {
            return None;
        }
        let st = self
            .state
            .entry(sig.instrument.clone())
            .or_insert_with(|| EmitterState {
                last_direction: SignalDirection::Flat,
                last_time: None,
                net_units: 0,
            });

        // Cooldown
        if let Some(t) = st.last_time {
            let elapsed = (sig.time - t).num_seconds();
            if elapsed < self.cfg.cooldown_secs {
                return None;
            }
        }

        // Decide intent based on signal direction vs current state.
        let units_delta: i64 = match sig.direction {
            SignalDirection::Long => {
                // Already at or above max long? Skip.
                if st.net_units >= self.cfg.max_position {
                    0
                } else if st.net_units < 0 {
                    // Flip from short → flatten + go long. Two-sided change.
                    -st.net_units + self.cfg.units_per_signal
                } else {
                    self.cfg.units_per_signal
                }
            }
            SignalDirection::Short => {
                if st.net_units <= -self.cfg.max_position {
                    0
                } else if st.net_units > 0 {
                    -st.net_units - self.cfg.units_per_signal
                } else {
                    -self.cfg.units_per_signal
                }
            }
            SignalDirection::Flat => {
                // Close any open position; do nothing if already flat.
                if st.net_units != 0 {
                    -st.net_units
                } else {
                    0
                }
            }
        };

        if units_delta == 0 {
            return None;
        }

        // Cap to max position after applying.
        let projected = st.net_units + units_delta;
        let units_delta = if projected.abs() > self.cfg.max_position {
            // Clamp to cap.
            let target = self.cfg.max_position * units_delta.signum();
            target - st.net_units
        } else {
            units_delta
        };

        if units_delta == 0 {
            return None;
        }

        // Update state.
        st.last_direction = sig.direction;
        st.last_time = Some(sig.time);
        // Don't update net_units yet — the router updates after fill.

        Some(EmittedIntent {
            instrument: sig.instrument.clone(),
            units_delta,
            triggered_by_model: sig.model_id.clone(),
        })
    }

    pub fn cfg(&self) -> &EmitterConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn sig(direction: SignalDirection, conf: f64, t: i64) -> StrategySignal {
        StrategySignal {
            instrument: "EUR_USD".into(),
            time: Utc.timestamp_opt(t, 0).unwrap(),
            direction,
            confidence: conf,
            probs: [0.0, 0.0, 0.0],
            model_id: "m1".into(),
            model_version: 1,
        }
    }

    #[test]
    fn below_threshold_no_emit() {
        let mut e = IntentEmitter::new();
        assert!(e.on_signal(&sig(SignalDirection::Long, 0.40, 0)).is_none());
    }

    #[test]
    fn long_then_short_flips_position() {
        let mut e = IntentEmitter::new();
        let i1 = e.on_signal(&sig(SignalDirection::Long, 0.7, 0)).unwrap();
        assert_eq!(i1.units_delta, 100);
        e.set_position("EUR_USD", 100);
        // Cooldown blocks immediate second emit.
        assert!(e.on_signal(&sig(SignalDirection::Short, 0.7, 5)).is_none());
        // After cooldown, flips: from +100 → close +100 then short 100 → -200 delta.
        let i2 = e.on_signal(&sig(SignalDirection::Short, 0.7, 60)).unwrap();
        assert_eq!(i2.units_delta, -200);
    }

    #[test]
    fn flat_signal_closes_position() {
        let mut e = IntentEmitter::new();
        e.set_position("EUR_USD", 300);
        let i = e.on_signal(&sig(SignalDirection::Flat, 0.8, 100)).unwrap();
        assert_eq!(i.units_delta, -300);
    }

    #[test]
    fn caps_at_max_position() {
        let cfg = EmitterConfig {
            max_position: 200,
            units_per_signal: 100,
            cooldown_secs: 0,
            confidence_threshold: 0.5,
        };
        let mut e = IntentEmitter::with_config(cfg);
        e.set_position("EUR_USD", 150);
        let i = e.on_signal(&sig(SignalDirection::Long, 0.7, 0)).unwrap();
        assert_eq!(
            i.units_delta, 50,
            "should clamp from 100 to 50 to hit max 200"
        );
    }
}

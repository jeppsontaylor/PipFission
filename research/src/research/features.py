"""On-the-fly feature recompute. Mirrors the Rust feature-engine's
output shape so the trained classifier inputs a 24-dim vector aligned
to what the live engine will feed at inference time.

Implementation note: this is a v1 port that uses pandas/numpy rolling
operations on bar-level OHLCV. The Rust live path does the same in
O(1) incremental state; the Python recompute is O(N) per refresh,
which is fine on a 1000-bar window. When we add the PyO3 wheel
(future milestone) this module will become a thin shim around the
Rust implementation, keeping research and live byte-for-byte aligned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


FEATURE_NAMES: list[str] = [
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",
    "log_ret_60",
    "vol_30",
    "vol_120",
    "vol_300",
    "spread_bp",
    "spread_z",
    "sma_dev_30",
    "sma_dev_120",
    "rsi_14",
    "bb_upper_dev",
    "bb_lower_dev",
    "atr_14",
    "macd",
    "macd_signal",
    "macd_hist",
    # Phase B re-engineering: `force_index_rel` replaces raw
    # `force_index` (= close-to-close-return × n_ticks). Raw values
    # ranged ±10 000 in active markets, dominating linear-model
    # gradients; rescaled by trailing 60-bar mean tick count it
    # collapses to ±50 while preserving sign + relative magnitude.
    "force_index_rel",
    "minute_sin",
    "minute_cos",
    # `log1p_n_ticks` replaces raw `n_ticks` (~0-10 000) with a
    # log-scaled version (~0-12). Same rank-ordering, much friendlier
    # to dashboard displays, logreg gradients, and StandardScaler
    # numerical conditioning.
    "log1p_n_ticks",
    "range_60",
    "drawdown_300",
]

assert len(FEATURE_NAMES) == 24


# Phase B retains the rolling window for `force_index_rel`'s denominator
# in this single named constant so the Rust mirror crate (bar-features)
# can match the value byte-for-byte. Don't change without updating
# `server/crates/bar-features/src/lib.rs`.
FORCE_INDEX_REL_WINDOW: int = 60


@dataclass(frozen=True)
class FeatureConfig:
    sma_short: int = 30
    sma_long: int = 120
    vol_short: int = 30
    vol_mid: int = 120
    vol_long: int = 300
    rsi_window: int = 14
    bb_window: int = 20
    bb_k: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    range_window: int = 60
    dd_window: int = 300


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return series.copy()
    alpha = 2.0 / (span + 1)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    # Wilder's RSI: simple moving average of gains / losses then ratio.
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    if len(close) >= window:
        avg_gain[:window] = np.mean(gain[:window])
        avg_loss[:window] = np.mean(loss[:window])
        for i in range(window, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
            avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(close), where=avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[avg_loss == 0] = 100.0
    return rsi / 100.0  # rescale to [0, 1] for stable model inputs.


def compute_features(bars: pl.DataFrame, cfg: FeatureConfig | None = None) -> pl.DataFrame:
    """Compute the 24-dim feature vector for every bar in `bars`.

    `bars` must be sorted ascending by `ts_ms` and have columns
    `[ts_ms, open, high, low, close, n_ticks, spread_bp_avg]`.
    Returns a polars DataFrame of length `len(bars)` with `ts_ms` and
    one column per feature name. Rows where the rolling stats haven't
    warmed up yet are still emitted (with safe-fallback values).
    """
    cfg = cfg or FeatureConfig()
    n = len(bars)
    if n == 0:
        return pl.DataFrame({name: [] for name in ["ts_ms"] + FEATURE_NAMES})

    close = bars["close"].to_numpy().astype(np.float64)
    high = bars["high"].to_numpy().astype(np.float64)
    low = bars["low"].to_numpy().astype(np.float64)
    spread_bp = bars["spread_bp_avg"].to_numpy().astype(np.float64)
    n_ticks = bars["n_ticks"].to_numpy().astype(np.float64)
    ts_ms = bars["ts_ms"].to_numpy().astype(np.int64)

    log_close = np.log(np.maximum(close, 1e-12))

    def lagged_logret(k: int) -> np.ndarray:
        out = np.zeros(n)
        if n > k:
            out[k:] = log_close[k:] - log_close[:-k]
        return out

    log_ret_1 = lagged_logret(1)
    log_ret_5 = lagged_logret(5)
    log_ret_20 = lagged_logret(20)
    log_ret_60 = lagged_logret(60)

    def rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
        if w <= 1 or n < w:
            return np.zeros(n)
        s = pl.Series(arr).rolling_std(w, min_samples=w)
        return s.to_numpy().astype(np.float64).copy()

    vol_30 = rolling_std(log_ret_1, cfg.vol_short)
    vol_120 = rolling_std(log_ret_1, cfg.vol_mid)
    vol_300 = rolling_std(log_ret_1, cfg.vol_long)

    spread_mean = pl.Series(spread_bp).rolling_mean(120, min_samples=20).to_numpy()
    spread_std = pl.Series(spread_bp).rolling_std(120, min_samples=20).to_numpy()
    spread_z = np.where(spread_std > 0, (spread_bp - spread_mean) / spread_std, 0.0)
    spread_z = np.nan_to_num(spread_z, nan=0.0)

    def sma_dev(window: int) -> np.ndarray:
        sma = pl.Series(close).rolling_mean(window, min_samples=window).to_numpy()
        out = np.zeros(n)
        mask = sma > 0
        out[mask] = close[mask] / sma[mask] - 1.0
        return out

    sma_dev_30 = sma_dev(cfg.sma_short)
    sma_dev_120 = sma_dev(cfg.sma_long)

    rsi_14 = _rsi(close, cfg.rsi_window)

    bb_mean = pl.Series(close).rolling_mean(cfg.bb_window, min_samples=cfg.bb_window).to_numpy()
    bb_std = pl.Series(close).rolling_std(cfg.bb_window, min_samples=cfg.bb_window).to_numpy()
    upper = bb_mean + cfg.bb_k * bb_std
    lower = bb_mean - cfg.bb_k * bb_std
    bb_upper_dev = np.where(close > 0, (close - upper) / close, 0.0)
    bb_lower_dev = np.where(close > 0, (close - lower) / close, 0.0)
    bb_upper_dev = np.nan_to_num(bb_upper_dev, nan=0.0)
    bb_lower_dev = np.nan_to_num(bb_lower_dev, nan=0.0)

    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr_14 = pl.Series(tr).rolling_mean(cfg.rsi_window, min_samples=cfg.rsi_window).to_numpy()
    atr_14 = np.nan_to_num(atr_14, nan=0.0)

    ema_fast = _ema(close, cfg.macd_fast)
    ema_slow = _ema(close, cfg.macd_slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, cfg.macd_signal)
    macd_hist = macd_line - macd_signal

    # Force index (Elder): close-to-close return × volume proxy.
    # Computed raw, then normalised by trailing-window mean n_ticks so
    # the magnitude stays bounded across regimes. The raw form swings
    # ±10 000 in active markets which dominates linear-model
    # gradients; the scaled form ranges ~±50 and survives Standard-
    # Scaler / dashboard display.
    force_index_raw = (close - prev_close) * n_ticks
    rolling_n_ticks_mean = (
        pl.Series(n_ticks)
        .rolling_mean(FORCE_INDEX_REL_WINDOW, min_samples=1)
        .to_numpy()
        .astype(np.float64)
    )
    force_index_rel = force_index_raw / np.maximum(rolling_n_ticks_mean, 1.0)
    # log1p of the raw tick count squashes the ~0-10 000 range into
    # ~0-10 (max(log(10001)) ≈ 9.21). Tree models keep the same split
    # information; linear models stop being driven by this one column.
    log1p_n_ticks = np.log1p(np.maximum(n_ticks, 0.0))

    # Cyclical encoding of bar's minute-within-hour (ts_ms is closed-bar
    # stamp = end of bucket, so this captures the bar's actual minute).
    minutes = (ts_ms // 60_000) % 60
    minute_sin = np.sin(2 * np.pi * minutes / 60)
    minute_cos = np.cos(2 * np.pi * minutes / 60)

    rolling_high = pl.Series(close).rolling_max(cfg.range_window, min_samples=cfg.range_window).to_numpy()
    rolling_low = pl.Series(close).rolling_min(cfg.range_window, min_samples=cfg.range_window).to_numpy()
    range_60 = np.where(close > 0, (rolling_high - rolling_low) / close, 0.0)
    range_60 = np.nan_to_num(range_60, nan=0.0)

    rolling_peak = pl.Series(close).rolling_max(cfg.dd_window, min_samples=cfg.dd_window).to_numpy()
    drawdown_300 = np.where(rolling_peak > 0, (close - rolling_peak) / rolling_peak, 0.0)
    drawdown_300 = np.nan_to_num(drawdown_300, nan=0.0)

    columns: dict[str, np.ndarray] = {
        "log_ret_1": log_ret_1,
        "log_ret_5": log_ret_5,
        "log_ret_20": log_ret_20,
        "log_ret_60": log_ret_60,
        "vol_30": np.nan_to_num(vol_30, nan=0.0),
        "vol_120": np.nan_to_num(vol_120, nan=0.0),
        "vol_300": np.nan_to_num(vol_300, nan=0.0),
        "spread_bp": spread_bp,
        "spread_z": spread_z,
        "sma_dev_30": np.nan_to_num(sma_dev_30, nan=0.0),
        "sma_dev_120": np.nan_to_num(sma_dev_120, nan=0.0),
        "rsi_14": rsi_14,
        "bb_upper_dev": bb_upper_dev,
        "bb_lower_dev": bb_lower_dev,
        "atr_14": atr_14,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "force_index_rel": force_index_rel,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
        "log1p_n_ticks": log1p_n_ticks,
        "range_60": range_60,
        "drawdown_300": drawdown_300,
    }
    out = pl.DataFrame({"ts_ms": ts_ms, **columns})
    return out

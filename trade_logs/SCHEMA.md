# `trade_logs/` schema cheatsheet

Single page, copy-pasteable. For the full README + query recipes see
[`README.md`](README.md).

All files are **newline-delimited JSON**, one record per line, capped
at the **2000 most-recent records** with atomic roll-off. Folder name
is the release version.

## Per-ticker files

| File | What |
| --- | --- |
| `<ticker>/trades.jsonl` | One closed round-trip per line |
| `<ticker>/actions.jsonl` | One `open_long`/`open_short`/`close` per line (skips excluded) |
| `<ticker>/skip_summary.jsonl` | One row per **run** of consecutive skips (compact) |
| `<ticker>/features.jsonl` | One per closed 10s bar: 24-dim feature dict + champion output |
| `<ticker>/summary.json` | Single-object snapshot — cum_R, hit rate, current model, latest probs |
| `<ticker>/LATEST.md` | Auto-rendered narrative — last 5 trades, last 5 actions, last 10 features |

## Cross-instrument files

| File | What |
| --- | --- |
| `training_log.jsonl` | One per pipeline run; carries the full model-zoo comparison + winner |
| `deployment_gate.jsonl` | One per quality-gate evaluation |
| `champion_changes.jsonl` | One per ONNX hot-swap |

## `trades.jsonl` record

```json
{
  "v": "v1.0.1",
  "instrument": "EUR_USD",
  "run_id": "live_20260428T134253Z",
  "ts_in_ms": 1777401900000,
  "ts_out_ms": 1777401990000,
  "side": 1,
  "qty": 1.0,
  "entry_px": 1.17161,
  "exit_px": 1.17160,
  "fee": 0.0,
  "slip": 0.0,
  "realized_r": -0.0000098,
  "net_pnl": -0.000012,
  "exit_reason": "stop_loss",
  "model_id": "side_lgbm_1777401438_5e7a5e",
  "model_kind": "lgbm",
  "params_id": "params_side_lgbm_1777401438_5e7a5e_860a5419",
  "n_features": 24,
  "model_oos_auc": 0.4355,
  "model_oos_log_loss": 0.7164,
  "entry_p_long": 0.5120,
  "entry_p_short": 0.4880,
  "entry_calibrated": 0.5120,
  "entry_spread_bp": 1.24,
  "entry_atr_14": 2.25e-05,
  "exit_p_long": 0.4821,
  "exit_p_short": 0.5179,
  "decision_chain": ["entry:signal", "exit:stop_loss"],
  "trader_params": { "long_threshold": 0.50, "max_hold_bars": 60, "..." : "..." },
  "snapshot_path": "./data/trades/<run_id>/<ts_out_ms>.json"
}
```

## `actions.jsonl` record

```json
{
  "v": "v1.0.1",
  "instrument": "EUR_USD",
  "ts_ms": 1777401900000,
  "action": "open_long",
  "reason": "signal",
  "price": 1.17161,
  "realized_r": null,
  "model_id": "side_lgbm_...",
  "params_id": "params_side_lgbm_..."
}
```

`action ∈ {open_long, open_short, close}`. `realized_r` is non-null
only on close events.

## `skip_summary.jsonl` record

```json
{
  "v": "v1.0.1",
  "instrument": "EUR_USD",
  "first_ts_ms": 1777401200000,
  "last_ts_ms": 1777401890000,
  "duration_ms": 690000,
  "count": 69,
  "by_reason": { "below_threshold": 60, "cooldown": 9 },
  "last_price": 1.1716,
  "model_id": "side_lgbm_...",
  "params_id": "params_side_lgbm_..."
}
```

A run flushes when (a) the next non-skip action arrives, or (b) 100
consecutive skips have built up. So one record covers an entire
period of inactivity, not one row per bar.

## `features.jsonl` record

```json
{
  "v": "v1.0.1",
  "instrument": "EUR_USD",
  "ts_ms": 1777401900000,
  "close": 1.171565,
  "n_ticks": 1,
  "spread_bp_avg": 1.28,
  "model_id": "side_lgbm_...",
  "kind": "onnx",
  "p_long": 0.4891,
  "p_short": 0.5109,
  "calibrated": 0.5109,
  "features": {
    "log_ret_1": -0.0,
    "log_ret_5": 0.0,
    "log_ret_20": 0.0,
    "log_ret_60": 0.0,
    "sma_dev_30": 0.0,
    "sma_dev_120": 0.0,
    "vol_30": 0.0,
    "vol_120": 0.0,
    "vol_300": 0.0,
    "rsi_14": 0.5,
    "atr_14": 0.0,
    "bb_upper_dev": 1.0,
    "bb_lower_dev": 1.0,
    "macd": 0.0,
    "macd_signal": 0.0,
    "macd_hist": 0.0,
    "range_60": 0.0,
    "drawdown_300": 0.0,
    "log1p_n_ticks": 0.69315,
    "force_index_rel": -0.0,
    "spread_bp": 1.28034,
    "spread_z": 0.0,
    "minute_sin": -1.0,
    "minute_cos": -0.0
  }
}
```

The 24 feature names live in `bar-features::FEATURE_NAMES`.
Values are rounded to 5 decimals to keep the JSONL line short.

## `summary.json` (per ticker)

```json
{
  "v": "v1.0.1",
  "instrument": "EUR_USD",
  "n_trades": 5,
  "wins": 1,
  "losses": 4,
  "hit_rate": 0.20,
  "cum_r": -0.000345,
  "by_exit_reason": { "stop_loss": 4, "take_profit": 1 },
  "latest_trade_ts_ms": 1777401990000,
  "latest_action": "close",
  "latest_close": 1.1716,
  "latest_p_long": 0.491,
  "latest_p_short": 0.509,
  "current_model_id": "side_lgbm_...",
  "summary_updated_ms": 1777402000000
}
```

## `training_log.jsonl` record

```json
{
  "v": "v1.0.1",
  "ts_ms": 1777400440443,
  "instrument": "EUR_USD",
  "run_id": "02a33092ab184bc4905e686c7fbeb3fb",
  "model_id": "side_lgbm_1777400246_ffb8d2",
  "model_kind": "lgbm",
  "n_features": 24,
  "n_train": 98,
  "n_oof": 490,
  "oos_auc": 0.4571,
  "oos_log_loss": 0.7068,
  "oos_brier": 0.2567,
  "oos_balanced_acc": 0.4681,
  "train_sortino": 0.1291,
  "fine_tune_sortino": 0.0,
  "fine_tune_max_dd_bp": 0.0,
  "passed_gate": false,
  "blocked_reasons": "oos_auc 0.457 < 0.550, ...",
  "lockbox_passed": false,
  "candidates": [
    { "spec": "lgbm",     "oos_log_loss": 0.706, "oos_auc": 0.457, "is_winner": true  },
    { "spec": "xgb",      "oos_log_loss": 0.711, "oos_auc": 0.443, "is_winner": false },
    { "spec": "catboost", "oos_log_loss": 0.715, "oos_auc": 0.439, "is_winner": false },
    { "spec": "logreg",   "oos_log_loss": 0.718, "oos_auc": 0.434, "is_winner": false },
    { "spec": "extratrees","oos_log_loss": 0.728, "oos_auc": 0.421, "is_winner": false },
    { "spec": "mlp",      "oos_log_loss": 0.722, "oos_auc": 0.428, "is_winner": false },
    { "spec": "histgb",   "oos_log_loss": 0.713, "oos_auc": 0.441, "is_winner": false }
  ],
  "params_id": "params_side_lgbm_1777400246_ffb8d2_a3314c17",
  "duration_secs": 145.32
}
```

## `deployment_gate.jsonl` record

```json
{
  "v": "v1.0.1",
  "ts_ms": 1777400440444,
  "instrument": "EUR_USD",
  "model_id": "side_lgbm_1777400246_ffb8d2",
  "passed": false,
  "blocked": [
    "oos_auc 0.457 < 0.550",
    "oos_log_loss 0.707 > 0.700",
    "oos_balanced_acc 0.468 < 0.520",
    "fine_tune_sortino 0.000 < 0.300",
    "lockbox did not pass"
  ],
  "thresholds": {
    "min_oos_auc": 0.55,
    "max_oos_log_loss": 0.70,
    "min_oos_balanced_acc": 0.52,
    "min_fine_tune_sortino": 0.30,
    "max_fine_tune_dd_bp": 1500.0,
    "require_lockbox_pass": true
  }
}
```

## `champion_changes.jsonl` record

```json
{
  "v": "v1.0.1",
  "ts_ms": 1777401438000,
  "new_model_id": "side_lgbm_1777401438_5e7a5e",
  "kind": "onnx",
  "n_features": 24
}
```

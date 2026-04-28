"""Spicy Penguin research layer.

Public sub-packages:
  research.data       — DuckDB extraction
  research.labeling   — wraps the Rust label_opt binary
  research.cv         — purged k-fold + CPCV wrappers (port of Rust crate)
  research.models     — model zoo + sigmoid calibration
  research.training   — Optuna driver + OOF parquet writer
  research.trader     — TraderParams + NSGA-II driver over trader_backtest
  research.stats      — Deflated Sharpe / PBO
  research.export     — ONNX + manifest writers
  research.lockbox    — single-shot 100-bar holdout gate
  research.cli        — `python -m research` subcommands
"""

__version__ = "0.1.0"

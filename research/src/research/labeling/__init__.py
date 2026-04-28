"""Wraps the Rust label_opt binary. The expensive interval-scheduling
optimiser stays in Rust; Python is just glue: extract bars, shell out,
and persist results to DuckDB.labels.
"""

from research.labeling.label_opt import (
    LabelOptConfig,
    run_label_opt,
    write_labels,
)

__all__ = ["LabelOptConfig", "run_label_opt", "write_labels"]

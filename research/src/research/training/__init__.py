"""Training orchestration: side classifier with purged CPCV → calibrated OOF."""

from research.training.side_train import train_side
from research.training.oof import OOF_SCHEMA, write_oof_parquet, write_oof_duckdb

__all__ = ["train_side", "OOF_SCHEMA", "write_oof_parquet", "write_oof_duckdb"]

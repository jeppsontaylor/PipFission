"""Canonical paths used across the research layer.

The repo root is auto-detected by walking up from this file until we
find a `Cargo.toml` (which lives at `server/Cargo.toml`) or a `.git`
directory. That makes `python -m research ...` work from any cwd.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        # `data/` always exists at the repo root in this project.
        if (parent / "data").is_dir() and (parent / "server").is_dir():
            return parent
    # Last resort: assume this file is at <repo>/research/src/research/paths.py
    return here.parents[3]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_dir: Path
    duckdb_path: Path
    server_dir: Path
    research_dir: Path
    artifacts_dir: Path
    venv_dir: Path

    @classmethod
    def detect(cls) -> "Paths":
        root = _find_repo_root()
        return cls(
            repo_root=root,
            data_dir=root / "data",
            duckdb_path=root / "data" / "oanda.duckdb",
            server_dir=root / "server",
            research_dir=root / "research",
            artifacts_dir=root / "research" / "artifacts",
            venv_dir=root / ".venv",
        )

    @property
    def label_opt_bin(self) -> Path:
        # Prefer the release build; fall back to debug if release doesn't exist.
        rel = self.server_dir / "target" / "release" / "label_opt"
        dbg = self.server_dir / "target" / "debug" / "label_opt"
        return rel if rel.exists() else dbg

    @property
    def trader_backtest_bin(self) -> Path:
        rel = self.server_dir / "target" / "release" / "trader_backtest"
        dbg = self.server_dir / "target" / "debug" / "trader_backtest"
        return rel if rel.exists() else dbg


PATHS = Paths.detect()

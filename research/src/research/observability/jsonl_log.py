"""Lightweight rolling JSONL writer for `trade_logs/<v>/...` previews.

Mirrors the Rust helper at
`server/crates/live-trader/src/jsonl_log.rs` so Python pipeline runs
write to the same agent-readable files.

- One record per line (newline-delimited JSON).
- After append, if the file has more than `MAX_LINES` lines, rewrite
  with the most recent `MAX_LINES` kept (atomic via tmp + rename).
- Heavy data still lives in DuckDB / `data/trades/` — these previews
  are for committed-to-git agent review only.
"""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

MAX_LINES: int = 2000
DEFAULT_ROOT_ENV = "TRADE_LOGS_DIR"
DEFAULT_ROOT = Path("./trade_logs")


def repo_version() -> str:
    """Read `version` from `server/Cargo.toml` so the Python side stays
    in lockstep with the Rust workspace version. Falls back to
    `PIPFISSION_VERSION` env, then `v0.0.0` if neither is available.
    """
    env_v = os.environ.get("PIPFISSION_VERSION")
    if env_v:
        return env_v if env_v.startswith("v") else f"v{env_v}"
    here = Path(__file__).resolve()
    # research/src/research/observability/jsonl_log.py → repo at parents[4]
    candidates = [
        here.parents[4] / "server" / "Cargo.toml",
        Path.cwd() / "server" / "Cargo.toml",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = tomllib.loads(p.read_text())
                v = data.get("workspace", {}).get("package", {}).get("version")
                if v:
                    return f"v{v}"
            except Exception:
                pass
    return "v0.0.0"


def trade_logs_root() -> Path:
    return Path(os.environ.get(DEFAULT_ROOT_ENV, str(DEFAULT_ROOT)))


def append(sub_path: str, record: dict[str, Any]) -> Path:
    """Append `record` as one JSONL line under
    `trade_logs/<version>/<sub_path>`. Creates parent dirs as needed.
    Returns the resolved path.
    """
    if ".." in sub_path.split("/") or sub_path.startswith("/"):
        raise ValueError(f"invalid sub_path: {sub_path!r}")
    version = repo_version()
    target = trade_logs_root() / version / sub_path
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), default=_jsonable) + "\n"
    with target.open("a", encoding="utf-8") as f:
        f.write(line)
    _roll_off_if_needed(target)
    return target


def _jsonable(o: Any) -> Any:
    """Best-effort conversion for objects json doesn't know about
    (numpy scalars, dataclasses, paths). Strings as last resort."""
    import dataclasses

    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _roll_off_if_needed(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    if len(lines) <= MAX_LINES:
        return
    kept = lines[-MAX_LINES:]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(kept)
    os.replace(tmp, path)

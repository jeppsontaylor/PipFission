"""Tests for `research.observability.jsonl_log`."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from research.observability.jsonl_log import (
    MAX_LINES,
    append,
    repo_version,
    trade_logs_root,
)


@pytest.fixture
def tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_LOGS_DIR", str(tmp_path))
    return tmp_path


def test_repo_version_reads_workspace_cargo_toml() -> None:
    v = repo_version()
    assert v.startswith("v"), v
    parts = v[1:].split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_append_creates_file_and_dirs(tmp_logs: Path) -> None:
    append("EUR_USD/trades.jsonl", {"a": 1, "b": "two"})
    target = tmp_logs / repo_version() / "EUR_USD" / "trades.jsonl"
    assert target.exists()
    line = target.read_text().splitlines()[0]
    rec = json.loads(line)
    assert rec == {"a": 1, "b": "two"}


def test_rolls_off_at_max_lines(tmp_logs: Path) -> None:
    for i in range(MAX_LINES + 25):
        append("EUR_USD/trades.jsonl", {"i": i})
    target = tmp_logs / repo_version() / "EUR_USD" / "trades.jsonl"
    n = sum(1 for _ in target.open())
    assert n == MAX_LINES
    first = json.loads(target.open().readline())
    assert first["i"] == 25  # we dropped the oldest 25


def test_rejects_traversal(tmp_logs: Path) -> None:
    with pytest.raises(ValueError):
        append("../escape.jsonl", {"x": 1})
    with pytest.raises(ValueError):
        append("/abs/path.jsonl", {"x": 1})


def test_trade_logs_root_respects_env(tmp_logs: Path, monkeypatch) -> None:
    custom = tmp_logs / "custom"
    monkeypatch.setenv("TRADE_LOGS_DIR", str(custom))
    assert trade_logs_root() == custom


def test_handles_path_objects(tmp_logs: Path) -> None:
    append("USD_JPY/decisions.jsonl", {"path": Path("/tmp/foo")})
    target = tmp_logs / repo_version() / "USD_JPY" / "decisions.jsonl"
    rec = json.loads(target.read_text())
    assert rec["path"] == "/tmp/foo"

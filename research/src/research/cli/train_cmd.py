"""`python -m research train` — side classifier training driver."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from research.observability import track_run
from research.training.side_train import SideTrainConfig, train_side

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def side(
    instrument: str = typer.Option(..., "--instrument", "-i"),
    n_bars: int = typer.Option(1000, "--n-bars"),
    n_splits: int = typer.Option(6, "--n-splits"),
    n_test_groups: int = typer.Option(2, "--n-test-groups"),
    embargo_pct: float = typer.Option(0.01, "--embargo-pct"),
    n_optuna_trials: int = typer.Option(25, "--trials"),
    label_run_id: Optional[str] = typer.Option(None, "--label-run-id"),
    seed: int = typer.Option(42, "--seed"),
    candidates: Optional[str] = typer.Option(
        None,
        "--candidates",
        help="CSV of model names to evaluate. Defaults to full zoo "
             "(lgbm,xgb,catboost,logreg,extratrees). Pick by OOS log loss.",
    ),
    json_out: Optional[Path] = typer.Option(
        None,
        "--json-out",
        help="If set, write the training result as a structured JSON "
             "object to this path. Used by the Rust pipeline-orchestrator "
             "to parse the `model_id` and per-candidate metrics without "
             "screen-scraping the rich-console output.",
    ),
) -> None:
    """Train the side classifier with purged CPCV and write OOF probs."""
    args = {
        "instrument": instrument,
        "n_bars": n_bars,
        "n_splits": n_splits,
        "n_test_groups": n_test_groups,
        "embargo_pct": embargo_pct,
        "n_optuna_trials": n_optuna_trials,
        "label_run_id": label_run_id,
        "seed": seed,
        "candidates": candidates,
    }
    with track_run("train.side", args, instrument=instrument):
        _run(instrument, n_bars, n_splits, n_test_groups, embargo_pct, n_optuna_trials, label_run_id, seed, candidates, json_out)


def _run(
    instrument: str,
    n_bars: int,
    n_splits: int,
    n_test_groups: int,
    embargo_pct: float,
    n_optuna_trials: int,
    label_run_id: Optional[str],
    seed: int,
    candidates: Optional[str],
    json_out: Optional[Path] = None,
) -> None:
    cfg = SideTrainConfig(
        instrument=instrument,
        n_bars=n_bars,
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo_pct=embargo_pct,
        n_optuna_trials=n_optuna_trials,
        label_run_id=label_run_id,
        seed=seed,
        candidates=candidates,
    )
    res = train_side(cfg)
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("model_id", res.model_id)
    table.add_row("spec_name", res.spec_name)
    table.add_row("instrument", res.instrument)
    table.add_row("n_train", str(res.n_train))
    table.add_row("n_oof", str(res.n_oof))
    table.add_row("oos_auc", f"{res.oos_auc:.4f}")
    table.add_row("oos_log_loss", f"{res.oos_log_loss:.4f}")
    table.add_row("oos_brier", f"{res.oos_brier:.4f}")
    table.add_row("oos_balanced_acc", f"{res.oos_balanced_acc:.4f}")
    table.add_row("train_sharpe", f"{res.train_sharpe:.4f}")
    table.add_row("max_train_sharpe", f"{res.max_train_sharpe:.4f}")
    table.add_row("train_sortino", f"{res.train_sortino:.4f}")
    table.add_row("max_train_sortino", f"{res.max_train_sortino:.4f}")
    table.add_row("oof_parquet", res.oof_parquet_path)
    table.add_row("label_run_id", res.label_run_id)
    console.print(table)

    # Show the candidate ranking so the operator can see what the zoo
    # tried and how the winner compared.
    cand_table = Table(title="Model zoo (sorted by OOS log loss, lower=better)")
    cand_table.add_column("spec")
    cand_table.add_column("OOS log loss", justify="right")
    cand_table.add_column("OOS AUC", justify="right")
    cand_table.add_column("OOS Brier", justify="right")
    cand_table.add_column("OOS bal-acc", justify="right")
    cand_table.add_column("n_oof", justify="right")
    cand_table.add_column("winner", justify="center")
    sorted_candidates = sorted(res.candidates, key=lambda c: c["oos_log_loss"])
    for c in sorted_candidates:
        cand_table.add_row(
            c["spec_name"],
            f"{c['oos_log_loss']:.4f}",
            f"{c['oos_auc']:.4f}",
            f"{c['oos_brier']:.4f}",
            f"{c['oos_balanced_acc']:.4f}",
            str(c["n_oof"]),
            "✓" if c["is_winner"] else "",
        )
    console.print(cand_table)

    # JSON-out for the Rust orchestrator. Same data the rich tables
    # show, just structured for machine consumption. Atomic via
    # tmp + rename so a partial write can't be parsed mid-flight.
    if json_out is not None:
        payload = asdict(res)
        # `fitted_on_full` (sklearn pipeline) isn't JSON-serialisable;
        # drop it — the Rust side only needs ids + metrics.
        payload.pop("fitted_on_full", None)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = json_out.with_suffix(json_out.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, default=str, indent=2))
        tmp.replace(json_out)
        console.print(f"[dim]wrote json_out to {json_out}[/dim]")

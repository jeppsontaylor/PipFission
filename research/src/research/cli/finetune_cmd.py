"""`python -m research finetune` — NSGA-II trader fine-tuner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from research.trader.optimizer import TraderFineTuneConfig, fine_tune_trader

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def run(
    instrument: str = typer.Option(..., "--instrument", "-i"),
    model_id: str = typer.Option(..., "--model-id"),
    n_train: int = typer.Option(1000, "--n-train"),
    n_fine_tune: int = typer.Option(100, "--n-fine-tune"),
    n_trials: int = typer.Option(50, "--trials"),
    cost_stress: float = typer.Option(1.0, "--cost-stress"),
    seed: int = typer.Option(7, "--seed"),
    json_out: Optional[Path] = typer.Option(
        None, "--json-out",
        help="If set, write the fine-tune report as structured JSON to "
             "this path. Read by the Rust pipeline-orchestrator.",
    ),
) -> None:
    """Fine-tune TraderParams on the next 100 unseen bars."""
    args = {
        "instrument": instrument,
        "model_id": model_id,
        "n_train": n_train,
        "n_fine_tune": n_fine_tune,
        "n_trials": n_trials,
        "cost_stress": cost_stress,
        "seed": seed,
    }
    _run(instrument, model_id, n_train, n_fine_tune, n_trials, cost_stress, seed, json_out)


def _run(
    instrument: str,
    model_id: str,
    n_train: int,
    n_fine_tune: int,
    n_trials: int,
    cost_stress: float,
    seed: int,
    json_out: Optional[Path] = None,
) -> None:
    cfg = TraderFineTuneConfig(
        instrument=instrument,
        model_id=model_id,
        n_train=n_train,
        n_fine_tune=n_fine_tune,
        n_trials=n_trials,
        seed=seed,
        cost_stress=cost_stress,
    )
    rep = fine_tune_trader(cfg)
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("study_id", rep["study_id"])
    table.add_row("model_id", rep["model_id"])
    table.add_row("params_id", rep["params_id"])
    table.add_row("n_trials", str(rep["n_trials"]))
    table.add_row("n_pareto", str(rep["n_pareto"]))
    in_s = rep["in_sample"]
    ft = rep["fine_tune"]
    table.add_row("in_sample_sharpe", f"{in_s.get('sharpe', 0):.4f}")
    table.add_row("in_sample_sortino", f"{in_s.get('sortino', 0):.4f}")
    table.add_row("fine_tune_sharpe", f"{ft.get('sharpe', 0):.4f}")
    table.add_row("fine_tune_sortino", f"{ft.get('sortino', 0):.4f}")
    table.add_row("max_dd_bp", f"{ft.get('max_drawdown_bp', 0):.2f}")
    table.add_row("turnover_per_day", f"{ft.get('turnover_per_day', 0):.4f}")
    table.add_row("hit_rate", f"{ft.get('hit_rate', 0):.4f}")
    table.add_row("n_trades", str(ft.get("n_trades", 0)))
    console.print(table)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = json_out.with_suffix(json_out.suffix + ".tmp")
        tmp.write_text(json.dumps(rep, default=str, indent=2))
        tmp.replace(json_out)
        console.print(f"[dim]wrote json_out to {json_out}[/dim]")

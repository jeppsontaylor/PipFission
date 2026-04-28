"""`python -m research pipeline run` — full retrain in one command.

Wraps `research.pipeline.run_pipeline` so the operator can invoke the
whole label → train → finetune → lockbox → export sequence with a
single CLI call. Each underlying step still records its own
`pipeline_runs` row, plus this command writes a parent
`pipeline.full` row so the dashboard can group them.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from research.pipeline import PipelineConfig, run_pipeline

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def run(
    instrument: str = typer.Option(..., "--instrument", "-i"),
    n_bars: int = typer.Option(1000, "--n-bars"),
    n_fine_tune: int = typer.Option(100, "--n-fine-tune"),
    n_optuna_trials: int = typer.Option(25, "--side-trials"),
    n_trader_trials: int = typer.Option(50, "--trader-trials"),
    seed: int = typer.Option(42, "--seed"),
    cost_stress: float = typer.Option(1.0, "--cost-stress"),
    min_dsr: float = typer.Option(0.50, "--min-dsr"),
    publish_on_lockbox_fail: bool = typer.Option(
        False,
        "--publish-on-lockbox-fail/--no-publish-on-lockbox-fail",
        help="Default false — a lockbox FAIL keeps the prior champion live.",
    ),
) -> None:
    """Run label → train → finetune → lockbox → export end-to-end."""
    cfg = PipelineConfig(
        instrument=instrument,
        n_bars=n_bars,
        n_fine_tune=n_fine_tune,
        n_optuna_trials=n_optuna_trials,
        n_trader_trials=n_trader_trials,
        seed=seed,
        cost_stress=cost_stress,
        min_dsr=min_dsr,
        publish_on_lockbox_fail=publish_on_lockbox_fail,
    )
    rep = run_pipeline(cfg)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("parent_run_id", rep.parent_run_id)
    table.add_row("instrument", rep.instrument)
    table.add_row("elapsed_ms", str(rep.elapsed_ms))
    table.add_row("label_run_id", rep.label_run_id or "—")
    table.add_row("n_labels", str(rep.n_labels))
    table.add_row("model_id", rep.model_id or "—")
    table.add_row("oos_auc", f"{rep.oos_auc:.4f}" if rep.oos_auc is not None else "—")
    table.add_row("params_id", rep.params_id or "—")
    table.add_row(
        "fine_tune_sortino",
        f"{rep.fine_tune_sortino:.4f}" if rep.fine_tune_sortino is not None else "—",
    )
    table.add_row(
        "fine_tune_max_dd_bp",
        f"{rep.fine_tune_max_dd_bp:.2f}" if rep.fine_tune_max_dd_bp is not None else "—",
    )
    table.add_row(
        "lockbox",
        "[green]PASS[/green]" if rep.lockbox_passed else "[red]FAIL[/red]" if rep.lockbox_passed is False else "—",
    )
    if rep.lockbox_reasons:
        table.add_row("reasons", "; ".join(rep.lockbox_reasons))
    table.add_row(
        "published",
        "[green]yes[/green]" if rep.published else "[yellow]no[/yellow]",
    )
    if rep.onnx_path:
        table.add_row("onnx_path", rep.onnx_path)
        table.add_row("sha256", (rep.onnx_sha256 or "")[:12] + "...")
    console.print(table)

    # Make the exit code reflect whether something blocking happened.
    # Convention: 0 = published or pipeline ran cleanly; 1 = lockbox blocked.
    if rep.lockbox_passed is False and not cfg.publish_on_lockbox_fail:
        raise typer.Exit(1)

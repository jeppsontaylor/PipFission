"""`python -m research lockbox` — single-shot 100-bar gate."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from research.lockbox.gate import LockboxConfig, seal_lockbox
from research.observability import track_run

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def seal(
    instrument: str = typer.Option(..., "--instrument", "-i"),
    model_id: str = typer.Option(..., "--model-id"),
    params_id: str = typer.Option(..., "--params-id"),
    n_seen: int = typer.Option(1100, "--n-seen"),
    n_lockbox: int = typer.Option(100, "--n-lockbox"),
    cost_stress: float = typer.Option(1.0, "--cost-stress"),
    min_n_trades: int = typer.Option(3, "--min-trades"),
    min_dsr: float = typer.Option(0.50, "--min-dsr"),
    max_dd_bp_limit: float = typer.Option(1500.0, "--max-dd-bp"),
    run_id: str = typer.Option(None, "--run-id"),
) -> None:
    """Run the lockbox evaluation. Single-shot per (model_id, params_id);
    re-running with the same run_id raises."""
    args = {
        "instrument": instrument,
        "model_id": model_id,
        "params_id": params_id,
        "n_seen": n_seen,
        "n_lockbox": n_lockbox,
        "cost_stress": cost_stress,
        "min_n_trades": min_n_trades,
        "min_dsr": min_dsr,
        "max_dd_bp_limit": max_dd_bp_limit,
        "run_id": run_id,
    }
    with track_run("lockbox", args, instrument=instrument):
        _run(instrument, model_id, params_id, n_seen, n_lockbox, cost_stress,
             min_n_trades, min_dsr, max_dd_bp_limit, run_id)


def _run(
    instrument: str,
    model_id: str,
    params_id: str,
    n_seen: int,
    n_lockbox: int,
    cost_stress: float,
    min_n_trades: int,
    min_dsr: float,
    max_dd_bp_limit: float,
    run_id: str,
) -> None:
    cfg = LockboxConfig(
        instrument=instrument, model_id=model_id, params_id=params_id,
        n_seen=n_seen, n_lockbox=n_lockbox, cost_stress=cost_stress,
        min_n_trades=min_n_trades, min_dsr=min_dsr, max_dd_bp_limit=max_dd_bp_limit,
    )
    res = seal_lockbox(cfg, run_id=run_id)
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("run_id", res.run_id)
    table.add_row("sealed", str(res.sealed))
    table.add_row("passed", "[green]YES[/green]" if res.passed else "[red]NO[/red]")
    s = res.summary
    table.add_row("n_trades", str(s.get("n_trades", 0)))
    table.add_row("net_return", f"{s.get('net_return', 0):.6f}")
    table.add_row("sharpe", f"{s.get('sharpe', 0):.4f}")
    table.add_row("sortino", f"{s.get('sortino', 0):.4f}")
    table.add_row("dsr", f"{s.get('dsr', 0):.4f}")
    table.add_row("max_dd_bp", f"{s.get('max_drawdown_bp', 0):.2f}")
    if res.reasons:
        table.add_row("reasons", "; ".join(res.reasons))
    console.print(table)
    if not res.passed:
        raise typer.Exit(1)

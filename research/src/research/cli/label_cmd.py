"""`python -m research label` — extract bars + run label optimiser."""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from research.data.extract import extract_bars_10s
from research.labeling.label_opt import LabelOptConfig, run_label_opt, write_labels
from research.observability import track_run

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def run(
    instrument: str = typer.Option(..., "--instrument", "-i", help="e.g. EUR_USD"),
    n_bars: int = typer.Option(1000, "--n-bars", help="Trailing-bar window size."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override the auto-generated run id."),
    sigma_span: int = typer.Option(60, help="EWMA σ span (bars)."),
    cusum_h_mult: float = typer.Option(1.0, help="CUSUM threshold = sigma * mult. Lower = more candidate events; 1.0 ≈ ~265 events on 1000 forex bars."),
    pt_atr: float = typer.Option(2.0, help="Profit-take ATR multiple."),
    sl_atr: float = typer.Option(2.0, help="Stop-loss ATR multiple."),
    vert_horizon: int = typer.Option(36, help="Vertical-barrier horizon (bars)."),
    min_edge: float = typer.Option(0.0, help="Per-trade edge floor (fractional return). 0 = keep all candidates; downstream cost model handles costs."),
    write: bool = typer.Option(True, help="Insert labels into DuckDB.labels."),
) -> None:
    """Optimise ideal buy/sell entries on the trailing N bars."""
    args = {
        "instrument": instrument,
        "n_bars": n_bars,
        "run_id": run_id,
        "sigma_span": sigma_span,
        "cusum_h_mult": cusum_h_mult,
        "pt_atr": pt_atr,
        "sl_atr": sl_atr,
        "vert_horizon": vert_horizon,
        "min_edge": min_edge,
        "write": write,
    }
    with track_run("label", args, instrument=instrument):
        _run(instrument, n_bars, run_id, sigma_span, cusum_h_mult, pt_atr, sl_atr, vert_horizon, min_edge, write)


def _run(
    instrument: str,
    n_bars: int,
    run_id: Optional[str],
    sigma_span: int,
    cusum_h_mult: float,
    pt_atr: float,
    sl_atr: float,
    vert_horizon: int,
    min_edge: float,
    write: bool,
) -> None:
    bars = extract_bars_10s(instrument=instrument, n_recent=n_bars)
    if bars.is_empty():
        console.print(f"[red]no bars in DuckDB for {instrument}[/red]")
        raise typer.Exit(2)
    console.print(f"loaded {len(bars)} bars for {instrument}")
    cfg = LabelOptConfig(
        sigma_span=sigma_span,
        cusum_h_mult=cusum_h_mult,
        pt_atr=pt_atr,
        sl_atr=sl_atr,
        vert_horizon=vert_horizon,
        min_edge=min_edge,
    )
    payload = run_label_opt(bars, cfg)
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("n_bars", str(payload["n_bars"]))
    table.add_row("n_events", str(payload["n_events"]))
    table.add_row("n_raw_labels", str(payload["n_raw_labels"]))
    table.add_row("n_chosen", str(len(payload["labels"])))
    if payload["labels"]:
        longs = sum(1 for r in payload["labels"] if r["side"] >= 0)
        shorts = sum(1 for r in payload["labels"] if r["side"] < 0)
        n = len(payload["labels"])
        minority = min(longs, shorts) / n if n else 0.0
        table.add_row("long/short", f"{longs}/{shorts}")
        table.add_row("minority frac", f"{minority:.2%}")
        avg_r = sum(abs(float(r["realized_r"])) for r in payload["labels"]) / n
        table.add_row("avg |realized_r|", f"{avg_r:.6f}")
    console.print(table)

    if write:
        rid = write_labels(instrument, payload, run_id)
        console.print(f"[green]wrote labels under run_id={rid}[/green]")
        # Also surface the run_id on stdout so the Rust orchestrator can
        # capture it without parsing the rich console output. One JSON
        # object per line is the same convention as the other CLIs'
        # --json-out flags but here we only need the id, so a single
        # marker line is enough.
        import json as _json
        print(_json.dumps({"label_run_id": rid, "n_chosen": len(payload.get("labels", []))}))

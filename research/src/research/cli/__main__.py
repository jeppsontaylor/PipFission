"""`python -m research <subcommand>` — Typer-driven CLI."""
from __future__ import annotations

import typer
from rich.console import Console

from research.cli import (
    export_cmd,
    finetune_cmd,
    train_cmd,
)

console = Console()
app = typer.Typer(
    add_completion=False,
    help="Spicy Penguin research layer: extract → label → train → fine-tune → lockbox → export.",
)

app.add_typer(train_cmd.app, name="train", help="Train the side classifier with purged CPCV.")
app.add_typer(finetune_cmd.app, name="finetune", help="NSGA-II trader optimiser on the next 100 bars.")
app.add_typer(export_cmd.app, name="export", help="Export champion to ONNX + manifest.")
# `pipeline` subcommand removed — the orchestrator is now the Rust
# binary `server/target/release/pipeline-orchestrator`. Each step
# (label / train / finetune / lockbox / export) is a separate CLI here
# and they thread IDs via --json-out files.


@app.command("init-schema")
def init_schema() -> None:
    """Apply the full DuckDB schema by shelling out to the Rust
    `init_schema` binary. Idempotent — safe to re-run."""
    import subprocess
    from research.paths import PATHS

    bin_path = PATHS.server_dir / "target" / "release" / "init_schema"
    if not bin_path.exists():
        bin_path = PATHS.server_dir / "target" / "debug" / "init_schema"
    if not bin_path.exists():
        console.print(
            "[red]init_schema binary missing; run "
            "`cargo build --release -p persistence --bin init_schema`[/red]"
        )
        raise typer.Exit(2)
    proc = subprocess.run(
        [str(bin_path), str(PATHS.duckdb_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    console.print(proc.stdout.strip())
    if proc.returncode != 0:
        console.print(f"[red]{proc.stderr.strip()}[/red]")
        raise typer.Exit(proc.returncode)


@app.command()
def info() -> None:
    """Report repo paths, DuckDB row counts, and Rust binary availability."""
    from research.paths import PATHS
    from research.data.duckdb_io import open_ro, instruments_with_data

    console.print(f"[bold]repo[/bold]      {PATHS.repo_root}")
    console.print(f"[bold]duckdb[/bold]    {PATHS.duckdb_path}")
    console.print(f"[bold]label_opt[/bold] {PATHS.label_opt_bin} {'OK' if PATHS.label_opt_bin.exists() else 'MISSING'}")
    console.print(f"[bold]trader_bt[/bold] {PATHS.trader_backtest_bin} {'OK' if PATHS.trader_backtest_bin.exists() else 'MISSING'}")

    if PATHS.duckdb_path.exists():
        c = open_ro()
        try:
            existing = {
                r[0]
                for r in c.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }
            for tbl in [
                "price_ticks", "bars_10s", "labels", "oof_predictions",
                "signals", "model_metrics", "trader_metrics",
                "optimizer_trials", "lockbox_results",
            ]:
                if tbl in existing:
                    n = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    console.print(f"  {tbl:<22} {n:>8,d}")
                else:
                    console.print(f"  {tbl:<22} [dim]—  (table missing; boot the Rust server to apply full schema)[/dim]")
        finally:
            c.close()
        try:
            instruments = instruments_with_data("bars_10s")
            console.print(f"  instruments in bars_10s: {instruments}")
        except Exception as exc:
            console.print(f"  instruments lookup failed: {exc}")
    else:
        console.print("[yellow]duckdb file does not exist yet — start the live server to create it[/yellow]")


if __name__ == "__main__":
    app()

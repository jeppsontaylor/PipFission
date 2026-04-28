"""`python -m research export` — ONNX export + manifest publish."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from research.export.manifest import write_manifest
from research.export.onnx_export import export_calibrated_to_onnx, verify_onnx_roundtrip
from research.observability import track_run

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def champion(
    model_id: str = typer.Option(..., "--model-id"),
    atol: float = typer.Option(1e-4, "--atol", help="Roundtrip tolerance."),
) -> None:
    """Convert champion to ONNX, verify, write manifest, publish to live dir."""
    with track_run("export.champion", {"model_id": model_id, "atol": atol}):
        _run(model_id, atol)


def _run(model_id: str, atol: float) -> None:
    info = export_calibrated_to_onnx(model_id)
    rt = verify_onnx_roundtrip(model_id, atol=atol)
    manifest = write_manifest(model_id, info["onnx_path"], info["feature_names"])
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("model_id", model_id)
    table.add_row("onnx_path", info["onnx_path"])
    table.add_row("sha256", manifest["sha256"][:12] + "...")
    table.add_row("n_features", str(info["n_features"]))
    table.add_row("roundtrip max_err", f"{rt['max_err']:.6e}")
    console.print(table)

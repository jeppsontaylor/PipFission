"""Allow `python -m research <subcommand>` by delegating to the Typer app."""
from research.cli.__main__ import app

if __name__ == "__main__":
    app()

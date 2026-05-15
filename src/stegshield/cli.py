from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Analyze image files for suspicious or dangerous indicators.")
console = Console()


@app.command()
def analyze(image_path: Path) -> None:
    """Analyze one image file."""
    if not image_path.exists():
        raise typer.BadParameter(f"File does not exist: {image_path}")

    console.print("[bold]StegShield analysis[/bold]")
    console.print(f"File: {image_path}")
    console.print("Status: metadata analyzer not implemented yet")


if __name__ == "__main__":
    app()

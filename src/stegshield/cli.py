from pathlib import Path
import json

import typer
from rich.console import Console
from rich.table import Table

from stegshield.analysis import analyze_image

app = typer.Typer(help="Analyze image files for suspicious or dangerous indicators.")
console = Console()


@app.callback()
def main() -> None:
    """Analyze image files for suspicious or dangerous indicators."""


@app.command()
def analyze(
    image_path: Path,
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON output."),
) -> None:
    """Analyze one image file."""
    if not image_path.exists():
        raise typer.BadParameter(f"File does not exist: {image_path}")
    if not image_path.is_file():
        raise typer.BadParameter(f"Path is not a file: {image_path}")

    result = analyze_image(image_path)

    if json_output:
        print(json.dumps(result, indent=2))
        return

    _print_human_report(result)


def _print_human_report(result: dict[str, object]) -> None:
    file_info = result["file"]
    risk = result["risk"]

    if not isinstance(file_info, dict) or not isinstance(risk, dict):
        raise RuntimeError("Invalid analysis result.")

    console.print("[bold]StegShield analysis[/bold]")
    console.print(f"File: {file_info['path']}")
    console.print(f"Label: [bold]{risk['label']}[/bold]")
    console.print(f"Risk score: {risk['risk_score']}")

    table = Table(title="File details")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("SHA-256", str(file_info["sha256"]))
    table.add_row("Detected type", str(file_info["detected_type"]))
    table.add_row("Detected MIME", str(file_info["detected_mime"]))
    table.add_row("Extension matches type", str(file_info["extension_matches_type"]))
    table.add_row("Size", f"{file_info['file_size_bytes']} bytes")
    table.add_row("Dimensions", _format_dimensions(file_info))
    table.add_row("Image mode", str(file_info["mode"]))
    table.add_row("Metadata fields", str(len(file_info["metadata_fields"])))
    console.print(table)

    indicators = risk["indicators"]
    if not indicators:
        console.print("[green]No suspicious indicators found.[/green]")
        return

    indicator_table = Table(title="Risk indicators")
    indicator_table.add_column("Severity")
    indicator_table.add_column("Code")
    indicator_table.add_column("Description")
    for indicator in indicators:
        indicator_table.add_row(
            str(indicator["severity"]),
            str(indicator["code"]),
            str(indicator["description"]),
        )
    console.print(indicator_table)


def _format_dimensions(file_info: dict[str, object]) -> str:
    width = file_info["width"]
    height = file_info["height"]
    if width is None or height is None:
        return "unknown"
    return f"{width}x{height}"


if __name__ == "__main__":
    app()

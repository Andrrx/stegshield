from pathlib import Path
import json

import typer
from rich.console import Console
from rich.table import Table

from stegshield.analysis import analyze_image
from stegshield.data.kaggle_stegoimages import create_kaggle_stegoimage_splits
from stegshield.data.splits import collect_labeled_images, create_stratified_splits, write_split_csvs

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


@app.command("prepare-dataset")
def prepare_dataset(
    raw_dir: Path = typer.Option(
        Path("data/raw"),
        "--raw-dir",
        help="Directory containing safe, suspicious, and dangerous subdirectories.",
    ),
    output_dir: Path = typer.Option(
        Path("data/splits"),
        "--output-dir",
        help="Directory where train.csv, val.csv, and test.csv will be written.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for deterministic splitting."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing train.csv, val.csv, and test.csv files.",
    ),
) -> None:
    """Create train/validation/test CSV files from labeled image folders."""
    existing_split_files = [
        output_dir / "train.csv",
        output_dir / "val.csv",
        output_dir / "test.csv",
    ]
    if not force and any(path.exists() for path in existing_split_files):
        raise typer.BadParameter(
            "Split CSV files already exist. Use --force only if you want to overwrite them."
        )

    samples = collect_labeled_images(raw_dir)
    if not samples:
        raise typer.BadParameter(f"No labeled images found under {raw_dir}")

    splits = create_stratified_splits(samples=samples, seed=seed)
    write_split_csvs(splits=splits, output_dir=output_dir)

    console.print("[bold]Dataset splits created[/bold]")
    console.print(f"Raw directory: {raw_dir}")
    console.print(f"Output directory: {output_dir}")
    for split_name, split_samples in splits.items():
        console.print(f"{split_name}: {len(split_samples)} samples")


@app.command("import-kaggle-splits")
def import_kaggle_splits(
    raw_dir: Path = typer.Option(
        Path("data/raw"),
        "--raw-dir",
        help="Project raw data directory containing labeled kaggle_stegoimages folders.",
    ),
    output_dir: Path = typer.Option(
        Path("data/splits"),
        "--output-dir",
        help="Directory where Kaggle split CSV files will be written.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing Kaggle split CSV files.",
    ),
) -> None:
    """Create split CSVs while preserving the Kaggle Stego Images official split."""
    try:
        splits = create_kaggle_stegoimage_splits(
            raw_dir=raw_dir,
            output_dir=output_dir,
            force=force,
        )
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print("[bold]Kaggle Stego Images splits imported[/bold]")
    console.print(f"Raw directory: {raw_dir}")
    console.print(f"Output directory: {output_dir}")
    for split_name, split_samples in splits.items():
        console.print(f"{split_name}: {len(split_samples)} samples")


@app.command("train-cnn")
def train_cnn_command(
    train_csv: Path = typer.Option(Path("data/splits/train.csv"), "--train-csv"),
    val_csv: Path = typer.Option(Path("data/splits/val.csv"), "--val-csv"),
    output_model: Path = typer.Option(Path("outputs/models/stegshield_cnn.pt"), "--output-model"),
    output_metrics: Path = typer.Option(
        Path("outputs/reports/training_metrics.json"),
        "--output-metrics",
    ),
    epochs: int = typer.Option(5, "--epochs", min=1),
    batch_size: int = typer.Option(16, "--batch-size", min=1),
    learning_rate: float = typer.Option(0.001, "--learning-rate", min=0.0),
    image_size: int = typer.Option(256, "--image-size", min=32),
    device: str = typer.Option("cpu", "--device", help="Torch device, for example cpu or cuda."),
) -> None:
    """Train the custom CNN from scratch using prepared split CSV files."""
    from stegshield.train_cnn import TrainingConfig, train_cnn

    config = TrainingConfig(
        train_csv=train_csv,
        val_csv=val_csv,
        output_model=output_model,
        output_metrics=output_metrics,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        device=device,
    )
    metrics = train_cnn(config)

    console.print("[bold]Training complete[/bold]")
    console.print(f"Model: {output_model}")
    console.print(f"Metrics: {output_metrics}")
    console.print(f"Best validation accuracy: {metrics['best_val_accuracy']:.4f}")


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

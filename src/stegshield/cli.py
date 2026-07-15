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
    cnn_model_path: Path | None = typer.Option(
        None,
        "--cnn-model-path",
        help="Optional binary clean/stego CNN checkpoint for hybrid fusion.",
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device for optional CNN inference."),
) -> None:
    """Analyze one image file."""
    if not image_path.exists():
        raise typer.BadParameter(f"File does not exist: {image_path}")
    if not image_path.is_file():
        raise typer.BadParameter(f"Path is not a file: {image_path}")

    result = analyze_image(image_path, cnn_model_path=cnn_model_path, device=device)

    if json_output:
        print(json.dumps(result, indent=2))
        return

    _print_human_report(result)


@app.command()
def scan(
    image_path: Path,
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON output."),
    cnn_model_path: Path | None = typer.Option(
        None,
        "--cnn-model-path",
        help="Hardened clean/stego CNN checkpoint (used only for lossless images).",
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device for optional CNN inference."),
) -> None:
    """Scan one image through the deployment router (format-aware defense layer).

    Lossless images (PNG/BMP/TIFF) get spatial CNN + static analysis; lossy images
    (JPEG/WebP) get static analysis only, since re-encoding neutralizes LSB stego.
    """
    if not image_path.exists():
        raise typer.BadParameter(f"File does not exist: {image_path}")
    if not image_path.is_file():
        raise typer.BadParameter(f"Path is not a file: {image_path}")

    from stegshield.router import scan_image

    predictor = None
    if cnn_model_path is not None:
        from stegshield.predict_cnn import StegoPredictor

        predictor = StegoPredictor(model_path=cnn_model_path, device=device)

    verdict = scan_image(image_path, predictor=predictor, device=device)

    if json_output:
        print(json.dumps(verdict.to_dict(), indent=2))
        return

    console.print("[bold]StegShield scan[/bold]")
    console.print(f"File: {verdict.path}")
    console.print(f"Detected type: {verdict.detected_type} ({verdict.processing_state})")
    console.print(f"Spatial LSB analysis applicable: {verdict.spatial_lsb_applicable}")
    console.print(f"Analyses run: {', '.join(verdict.analyses_run)}")
    console.print(f"Label: [bold]{verdict.label}[/bold]  (risk {verdict.risk_score})")
    if verdict.cnn_stego_probability is not None:
        console.print(f"CNN stego probability: {verdict.cnn_stego_probability:.4f}")
    console.print(f"Explanation: {verdict.explanation}")
    console.print(f"Latency: {verdict.latency_ms} ms")


@app.command("doctor")
def doctor(
    device: str = typer.Option("cuda", "--device", help="Device to validate, for example cuda or cuda:0."),
) -> None:
    """Print Python, PyTorch, and CUDA diagnostics."""
    from stegshield.doctor import build_torch_doctor_report

    report = build_torch_doctor_report(requested_device=device)
    table = Table(title="StegShield ML Doctor")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Python", report.python_version)
    table.add_row("Torch", report.torch_version)
    table.add_row("torch.cuda.is_available()", str(report.cuda_available))
    table.add_row("torch.version.cuda", str(report.torch_cuda_version))
    table.add_row("cuDNN enabled", str(report.cudnn_enabled))
    table.add_row("CUDA device count", str(report.cuda_device_count))
    table.add_row("CUDA device names", ", ".join(report.cuda_device_names) or "none")
    table.add_row("Current CUDA device", str(report.current_cuda_device))
    table.add_row("Current CUDA device name", str(report.current_cuda_device_name))
    table.add_row("Requested device", report.requested_device)
    table.add_row("Requested device valid", str(report.requested_device_valid))
    console.print(table)
    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


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


@app.command("make-payload-regression-set")
def make_payload_regression_set_command(
    train_csv: Path = typer.Option(
        Path("data/splits/train.csv"),
        "--train-csv",
        help="Kaggle train split; its clean images source regress_train and regress_val.",
    ),
    test_csv: Path = typer.Option(
        Path("data/splits/test_standard.csv"),
        "--test-csv",
        help="Kaggle test split; its clean images source regress_test (no source leakage).",
    ),
    output_image_dir: Path = typer.Option(
        Path("data/processed/regress"),
        "--output-image-dir",
        help="Directory where synthetic embedded PNGs are written.",
    ),
    output_split_dir: Path = typer.Option(
        Path("data/splits"),
        "--output-split-dir",
        help="Directory where regress_{train,val,test}.csv are written.",
    ),
    image_size: int = typer.Option(256, "--image-size", min=32, help="Crop size for generated images."),
    val_fraction: float = typer.Option(0.15, "--val-fraction", min=0.0, max=0.9),
    clean_fraction: float = typer.Option(
        0.2,
        "--clean-fraction",
        min=0.0,
        max=1.0,
        help="Fraction of generated images left un-embedded (payload size unknown/masked).",
    ),
    seed: int = typer.Option(1234, "--seed", help="Deterministic payload-size sampling seed."),
    limit_per_split: int | None = typer.Option(
        None,
        "--limit-per-split",
        min=1,
        help="Optional cap on generated images per split, for quick experiments.",
    ),
) -> None:
    """Generate a synthetic sequential-LSB set with known payload sizes for regression.

    Payloads are inert os.urandom bytes embedded with StegShield's own embedder, so the
    CNN regressor learns from independent ground truth rather than from the statistical
    estimator's outputs (avoids circularity in the CNN-vs-statistical comparison).
    """
    from stegshield.data.synth_lsb import RegressionSetConfig, make_payload_regression_set

    splits = make_payload_regression_set(
        RegressionSetConfig(
            train_csv=train_csv,
            test_csv=test_csv,
            output_image_dir=output_image_dir,
            output_split_dir=output_split_dir,
            image_size=image_size,
            val_fraction=val_fraction,
            clean_fraction=clean_fraction,
            seed=seed,
            limit_per_split=limit_per_split,
        )
    )
    console.print("[bold]Payload regression set created[/bold]")
    console.print(f"Image directory: {output_image_dir}")
    console.print(f"Split directory: {output_split_dir}")
    for split_name, samples in splits.items():
        embedded = sum(1 for sample in samples if sample.payload_bytes is not None)
        console.print(
            f"regress_{split_name}: {len(samples)} images ({embedded} embedded, "
            f"{len(samples) - embedded} clean)"
        )


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
    weight_decay: float = typer.Option(0.0001, "--weight-decay", min=0.0),
    image_size: int = typer.Option(256, "--image-size", min=32),
    device: str = typer.Option("cpu", "--device", help="Torch device, for example cpu or cuda."),
    model_name: str = typer.Option(
        "steganalysis",
        "--model",
        help="CNN model variant: yedroudj (Yedroudj-Net baseline) or steganalysis.",
    ),
    normalization: str = typer.Option(
        "raw255",
        "--normalization",
        help="Input normalization mode: raw255 (0-255 pixels, recommended), none, or imagenet.",
    ),
    crop: str = typer.Option(
        "top-left",
        "--crop",
        help="Crop position: top-left (keeps sequential LSB payload region) or center.",
    ),
    task: str = typer.Option(
        "stego",
        "--task",
        help="CNN training task: stego for clean/stego or risk for safe/suspicious/dangerous.",
    ),
    class_weights: bool = typer.Option(
        True,
        "--class-weights/--no-class-weights",
        help="Use inverse-frequency class weights in CrossEntropyLoss.",
    ),
    balanced_sampler: bool = typer.Option(
        True,
        "--balanced-sampler/--no-balanced-sampler",
        help="Sample training rows with inverse-frequency weights to reduce class imbalance.",
    ),
    selection_metric: str = typer.Option(
        "macro_f1",
        "--selection-metric",
        help="Validation metric used for best checkpoint selection: macro_f1 or balanced_accuracy.",
    ),
    num_workers: int = typer.Option(
        0,
        "--num-workers",
        min=0,
        help="DataLoader worker processes. 4-8 recommended; PNG decoding is the bottleneck at 0.",
    ),
    amp: bool = typer.Option(
        False,
        "--amp/--no-amp",
        help="Mixed-precision training on CUDA: roughly halves VRAM use and speeds up training.",
    ),
    payload_head: bool = typer.Option(
        False,
        "--payload-head/--no-payload-head",
        help="Add a payload-size regression head (steganalysis model only); needs payload_bytes labels.",
    ),
    payload_loss_weight: float = typer.Option(
        0.5,
        "--payload-loss-weight",
        min=0.0,
        help="Weight of the masked smooth-L1 payload-regression loss in the total loss.",
    ),
    augment: bool = typer.Option(
        False,
        "--augment/--no-augment",
        help="Payload-preserving processing augmentation (resize/blur/noise/re-save) for robustness.",
    ),
) -> None:
    """Train the custom CNN from scratch using prepared split CSV files."""
    from stegshield.doctor import build_torch_doctor_report
    from stegshield.train_cnn import TrainingConfig, train_cnn

    config = TrainingConfig(
        train_csv=train_csv,
        val_csv=val_csv,
        output_model=output_model,
        output_metrics=output_metrics,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        image_size=image_size,
        device=device,
        model_name=model_name,
        normalization=normalization,
        crop=crop,
        task=task,
        class_weights=class_weights,
        balanced_sampler=balanced_sampler,
        selection_metric=selection_metric,
        num_workers=num_workers,
        amp=amp,
        payload_head=payload_head,
        payload_loss_weight=payload_loss_weight,
        augment=augment,
    )
    doctor_report = build_torch_doctor_report(requested_device=device)
    console.print("[bold]Training device[/bold]")
    console.print(f"Requested device: {device}")
    console.print(f"CUDA available: {doctor_report.cuda_available}")
    console.print(f"Torch CUDA version: {doctor_report.torch_cuda_version}")
    console.print(f"cuDNN enabled: {doctor_report.cudnn_enabled}")
    console.print(f"CUDA devices: {doctor_report.cuda_device_names or ['none']}")
    console.print(f"num_workers: {num_workers}")
    for warning in doctor_report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    metrics = train_cnn(config)

    console.print("[bold]Training complete[/bold]")
    console.print(f"Model: {output_model}")
    console.print(f"Metrics: {output_metrics}")
    console.print(
        f"Best {metrics['best_selection_metric_name']}: "
        f"{metrics['best_selection_metric']:.4f} at epoch {metrics['best_epoch']}"
    )
    console.print(f"Best validation accuracy: {metrics['best_val_accuracy']:.4f}")
    if payload_head:
        best_val = metrics.get("best_val_metrics") or {}
        mae = best_val.get("payload_mae_bytes") if isinstance(best_val, dict) else None
        if mae is not None:
            console.print(f"Best-epoch validation payload MAE: {mae} bytes")


@app.command("evaluate-cnn")
def evaluate_cnn_command(
    model_path: Path = typer.Option(Path("outputs/models/stegshield_cnn.pt"), "--model-path"),
    split_csv: Path = typer.Option(Path("data/splits/test_standard.csv"), "--split-csv"),
    output_report: Path = typer.Option(
        Path("outputs/reports/evaluation_metrics.json"),
        "--output-report",
    ),
    raw_dir: Path | None = typer.Option(
        None,
        "--raw-dir",
        help="Override raw data directory for resolving relative split paths.",
    ),
    batch_size: int = typer.Option(16, "--batch-size", min=1),
    device: str = typer.Option("cpu", "--device", help="Torch device, for example cpu or cuda."),
    num_workers: int = typer.Option(
        0,
        "--num-workers",
        min=0,
        help="DataLoader worker processes. Use 0 on Windows if worker startup is unstable.",
    ),
    model_name: str | None = typer.Option(
        None,
        "--model",
        help="Override checkpoint model variant: yedroudj or steganalysis.",
    ),
    image_size: int | None = typer.Option(
        None,
        "--image-size",
        min=32,
        help="Override checkpoint image size.",
    ),
    normalization: str | None = typer.Option(
        None,
        "--normalization",
        help="Override checkpoint normalization mode: raw255, none, or imagenet.",
    ),
    crop: str | None = typer.Option(
        None,
        "--crop",
        help="Override checkpoint crop position: top-left or center.",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help="Override checkpoint task: stego or risk.",
    ),
) -> None:
    """Evaluate a trained CNN checkpoint on one split CSV."""
    from stegshield.evaluate_cnn import EvaluationConfig, evaluate_cnn

    config = EvaluationConfig(
        model_path=model_path,
        split_csv=split_csv,
        output_report=output_report,
        raw_dir=raw_dir,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        model_name=model_name,
        image_size=image_size,
        normalization=normalization,
        crop=crop,
        task=task,
    )
    report = evaluate_cnn(config)

    console.print("[bold]Evaluation complete[/bold]")
    console.print(f"Model: {model_path}")
    console.print(f"Split: {split_csv}")
    console.print(f"Report: {output_report}")
    console.print(f"Accuracy: {report['accuracy']:.4f}")
    console.print(f"Macro F1: {report['macro_f1']:.4f}")


@app.command("evaluate-fusion")
def evaluate_fusion_command(
    model_path: Path = typer.Option(Path("outputs/models/stegshield_cnn.pt"), "--model-path"),
    split_csv: Path = typer.Option(Path("data/splits/test_standard.csv"), "--split-csv"),
    output_report: Path = typer.Option(
        Path("outputs/reports/fusion_evaluation_metrics.json"),
        "--output-report",
    ),
    raw_dir: Path | None = typer.Option(
        None,
        "--raw-dir",
        help="Override raw data directory for resolving relative split paths.",
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device, for example cpu or cuda."),
    cnn_suspicious_threshold: float = typer.Option(
        0.3,
        "--cnn-suspicious-threshold",
        min=0.0,
        max=1.0,
        help="CNN-only stego probability threshold for suspicious risk.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Optional maximum number of split rows to evaluate.",
    ),
    payload_source: str = typer.Option(
        "statistical",
        "--payload-source",
        help="LSB payload severity source: statistical (default), cnn (needs --payload-head model), or both.",
    ),
) -> None:
    """Evaluate metadata-only, CNN-only, and fused final risk classification."""
    from stegshield.evaluate_fusion import FusionEvaluationConfig, evaluate_fusion

    config = FusionEvaluationConfig(
        model_path=model_path,
        split_csv=split_csv,
        output_report=output_report,
        raw_dir=raw_dir,
        device=device,
        cnn_suspicious_threshold=cnn_suspicious_threshold,
        limit=limit,
        payload_source=payload_source,
    )
    report = evaluate_fusion(config)

    console.print("[bold]Fusion evaluation complete[/bold]")
    console.print(f"Model: {model_path}")
    console.print(f"Split: {split_csv}")
    console.print(f"Report: {output_report}")
    console.print(f"Payload source: {payload_source}")
    console.print(f"Fused accuracy: {report['methods']['fused']['accuracy']:.4f}")
    console.print(f"Fused macro F1: {report['methods']['fused']['macro_f1']:.4f}")


@app.command("evaluate-robustness")
def evaluate_robustness_command(
    model_path: Path = typer.Option(Path("outputs/models/steganalysis_stego.pt"), "--model-path"),
    split_csv: Path = typer.Option(Path("data/splits/test_standard.csv"), "--split-csv"),
    output_report: Path = typer.Option(
        Path("outputs/reports/robustness_benchmark.json"),
        "--output-report",
    ),
    raw_dir: Path | None = typer.Option(None, "--raw-dir"),
    device: str = typer.Option("cpu", "--device"),
    threshold: float = typer.Option(0.5, "--threshold", min=0.0, max=1.0),
    limit_per_class: int | None = typer.Option(
        200,
        "--limit-per-class",
        min=1,
        help="Max clean and max stego images sampled per operation (speed).",
    ),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
) -> None:
    """Measure detection under benign and lossy processing (JPEG, resize, blur, noise)."""
    from stegshield.robustness import RobustnessConfig, evaluate_robustness

    report = evaluate_robustness(
        RobustnessConfig(
            model_path=model_path,
            split_csv=split_csv,
            output_report=output_report,
            raw_dir=raw_dir,
            device=device,
            threshold=threshold,
            limit_per_class=limit_per_class,
            batch_size=batch_size,
        )
    )
    console.print("[bold]Robustness benchmark complete[/bold]")
    console.print(f"Report: {output_report}")
    console.print(f"{'operation':14s} {'detect':>8s} {'clean_FPR':>10s}")
    for op in report["operations"]:
        console.print(
            f"{op['name']:14s} {op['stego_detection_rate']:>8.3f} {op['clean_fpr']:>10.3f}"
            + ("  [lossy]" if op["lossy"] else "")
        )


@app.command("evaluate-payload-regression")
def evaluate_payload_regression_command(
    model_path: Path = typer.Option(Path("outputs/models/steganalysis_multitask.pt"), "--model-path"),
    split_csv: Path = typer.Option(Path("data/splits/regress_test.csv"), "--split-csv"),
    output_report: Path = typer.Option(
        Path("outputs/reports/payload_regression_test.json"),
        "--output-report",
    ),
    raw_dir: Path | None = typer.Option(None, "--raw-dir"),
    batch_size: int = typer.Option(16, "--batch-size", min=1),
    device: str = typer.Option("cpu", "--device"),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
) -> None:
    """Evaluate CNN payload-size regression (MAE / median absolute error in bytes)."""
    from stegshield.payload_eval import PayloadRegressionConfig, evaluate_payload_regression

    report = evaluate_payload_regression(
        PayloadRegressionConfig(
            model_path=model_path,
            split_csv=split_csv,
            output_report=output_report,
            raw_dir=raw_dir,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )
    )
    console.print("[bold]Payload regression evaluation complete[/bold]")
    console.print(f"Report: {output_report}")
    console.print(f"Supervised samples: {report['supervised_count']}")
    console.print(f"MAE: {report['mae_bytes']} bytes")
    console.print(f"Median absolute error: {report['median_absolute_error_bytes']} bytes")


@app.command("evaluate-payload-agreement")
def evaluate_payload_agreement_command(
    model_path: Path = typer.Option(Path("outputs/models/steganalysis_multitask.pt"), "--model-path"),
    split_csv: Path = typer.Option(Path("data/splits/test_standard.csv"), "--split-csv"),
    output_report: Path = typer.Option(
        Path("outputs/reports/payload_agreement_test_standard.json"),
        "--output-report",
    ),
    raw_dir: Path | None = typer.Option(None, "--raw-dir"),
    batch_size: int = typer.Option(16, "--batch-size", min=1),
    device: str = typer.Option("cpu", "--device"),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    stego_threshold: float = typer.Option(0.5, "--stego-threshold", min=0.0, max=1.0),
) -> None:
    """Compare CNN payload estimates against the statistical estimator on real stego images."""
    from stegshield.payload_eval import PayloadAgreementConfig, evaluate_payload_agreement

    report = evaluate_payload_agreement(
        PayloadAgreementConfig(
            model_path=model_path,
            split_csv=split_csv,
            output_report=output_report,
            raw_dir=raw_dir,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            stego_threshold=stego_threshold,
        )
    )
    console.print("[bold]Payload agreement evaluation complete[/bold]")
    console.print(f"Report: {output_report}")
    console.print(f"Compared stego samples: {report['compared_count']}")
    console.print(f"Pearson (log2): {report['pearson_log2']}")
    console.print(f"Spearman (log2): {report['spearman_log2']}")


@app.command("summarize-experiments")
def summarize_experiments_command(
    report_paths: list[Path] = typer.Argument(
        ...,
        help="CNN or fusion evaluation JSON reports to combine.",
    ),
    output_path: Path = typer.Option(
        Path("outputs/reports/experiment_summary.md"),
        "--output",
        help="Markdown or JSON summary output path.",
    ),
    output_format: str = typer.Option(
        "markdown",
        "--format",
        help="Summary format: markdown or json.",
    ),
) -> None:
    """Combine evaluation JSON reports into a thesis-friendly summary table."""
    from stegshield.experiment_summary import ExperimentSummaryConfig, summarize_experiments

    rows = summarize_experiments(
        ExperimentSummaryConfig(
            report_paths=tuple(report_paths),
            output_path=output_path,
            output_format=output_format,
        )
    )
    console.print("[bold]Experiment summary written[/bold]")
    console.print(f"Output: {output_path}")
    console.print(f"Rows: {len(rows)}")


@app.command("plot-results")
def plot_results_command(
    report_paths: list[Path] = typer.Argument(
        ...,
        help="Training, CNN evaluation, or fusion evaluation JSON reports.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/figures"),
        "--output-dir",
        help="Directory where figure PNG files are written.",
    ),
    dpi: int = typer.Option(300, "--dpi", min=72, help="Figure resolution for thesis export."),
) -> None:
    """Create thesis figures: training curves, confusion matrices, ROC, comparison chart."""
    from stegshield.plots import PlotConfig, generate_plots

    for report_path in report_paths:
        if not report_path.exists():
            raise typer.BadParameter(f"Report does not exist: {report_path}")

    written = generate_plots(
        PlotConfig(report_paths=tuple(report_paths), output_dir=output_dir, dpi=dpi)
    )
    console.print("[bold]Figures written[/bold]")
    for path in written:
        console.print(str(path))


def _print_human_report(result: dict[str, object]) -> None:
    file_info = result["file"]
    risk = result["risk"]

    if not isinstance(file_info, dict) or not isinstance(risk, dict):
        raise RuntimeError("Invalid analysis result.")

    console.print("[bold]StegShield analysis[/bold]")
    console.print(f"File: {file_info['path']}")
    console.print(f"Label: [bold]{risk['label']}[/bold]")
    console.print(f"Risk score: {risk['risk_score']}")
    if "fusion" in result:
        fusion = result["fusion"]
        if isinstance(fusion, dict):
            console.print(f"Fused label: [bold]{fusion['label']}[/bold]")
            console.print(f"Fused risk score: {fusion['risk_score']}")
            console.print(f"CNN stego probability: {fusion['cnn_stego_probability']}")

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

from typer.testing import CliRunner
from PIL import Image

import pytest

from stegshield.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Analyze image files" in result.output


def test_analyze_safe_png(tmp_path) -> None:
    image_path = tmp_path / "safe.png"
    Image.new("RGB", (10, 10), color="white").save(image_path)

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(image_path), "--json"])

    assert result.exit_code == 0
    assert '"label": "safe"' in result.output
    assert '"detected_type": "png"' in result.output


def test_analyze_extension_mismatch(tmp_path) -> None:
    image_path = tmp_path / "mismatch.jpg"
    Image.new("RGB", (10, 10), color="white").save(image_path, format="PNG")

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(image_path), "--json"])

    assert result.exit_code == 0
    assert '"label": "suspicious"' in result.output
    assert "extension_type_mismatch" in result.output


def test_prepare_dataset_does_not_overwrite_existing_splits_without_force(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    safe_dir = raw_dir / "safe"
    safe_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10), color="white").save(safe_dir / "safe.png")

    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    (output_dir / "train.csv").write_text("path,label\nexisting.png,safe\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "prepare-dataset",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "already exist" in result.output
    assert (output_dir / "train.csv").read_text(encoding="utf-8") == (
        "path,label\nexisting.png,safe\n"
    )


def test_analyze_outputs_optional_cnn_fusion(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from stegshield.labels import STEGO_LABELS
    from stegshield.models.cnn import create_cnn_model

    image_path = tmp_path / "safe.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    model = create_cnn_model("yedroudj", num_classes=len(STEGO_LABELS))
    model_path = tmp_path / "binary_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": 16,
            "model_name": "yedroudj",
            "normalization": "none",
            "task": "stego",
        },
        model_path,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analyze",
            str(image_path),
            "--json",
            "--cnn-model-path",
            str(model_path),
        ],
    )

    assert result.exit_code == 0
    assert '"risk"' in result.output
    assert '"fusion"' in result.output
    assert '"cnn_stego_probability"' in result.output

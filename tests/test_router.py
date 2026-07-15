from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stegshield.data.synth_lsb import embed_sequential_lsb  # noqa: E402
from stegshield.labels import STEGO_LABELS  # noqa: E402
from stegshield.models.cnn import create_cnn_model  # noqa: E402
from stegshield.predict_cnn import StegoPredictor  # noqa: E402
from stegshield.router import processing_state, scan_image  # noqa: E402


def test_processing_state_mapping() -> None:
    assert processing_state("png") == ("lossless", True)
    assert processing_state("bmp") == ("lossless", True)
    assert processing_state("jpeg") == ("lossy", False)
    assert processing_state("webp") == ("lossy", False)
    assert processing_state("unknown") == ("unknown", False)


def _predictor(tmp_path: Path, image_size: int) -> StegoPredictor:
    model = create_cnn_model("steganalysis", num_classes=len(STEGO_LABELS))
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": image_size,
            "model_name": "steganalysis",
            "normalization": "raw255",
            "crop": "top-left",
            "task": "stego",
        },
        path,
    )
    return StegoPredictor(model_path=path, device="cpu")


def test_scan_lossless_png_runs_spatial_cnn(tmp_path: Path) -> None:
    image_size = 64
    base = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    png = tmp_path / "stego.png"
    Image.fromarray(embed_sequential_lsb(base, b"\x03" * 200), "RGB").save(png)

    verdict = scan_image(png, predictor=_predictor(tmp_path, image_size))

    assert verdict.processing_state == "lossless"
    assert verdict.spatial_lsb_applicable is True
    assert "spatial_cnn" in verdict.analyses_run
    assert verdict.cnn_stego_probability is not None
    assert verdict.label in {"safe", "suspicious", "dangerous"}


def test_scan_jpeg_is_static_only(tmp_path: Path) -> None:
    image_size = 64
    jpg = tmp_path / "photo.jpg"
    Image.new("RGB", (image_size, image_size), "white").save(jpg, format="JPEG", quality=90)

    verdict = scan_image(jpg, predictor=_predictor(tmp_path, image_size))

    assert verdict.processing_state == "lossy"
    assert verdict.spatial_lsb_applicable is False
    assert "spatial_cnn" not in verdict.analyses_run
    assert verdict.cnn_stego_probability is None
    assert "not applicable" in verdict.explanation


def test_scan_without_model_is_static_only(tmp_path: Path) -> None:
    image_size = 64
    png = tmp_path / "clean.png"
    Image.new("RGB", (image_size, image_size), "white").save(png)

    verdict = scan_image(png, predictor=None)

    assert verdict.cnn_stego_probability is None
    assert "spatial_cnn" not in verdict.analyses_run
    assert verdict.label == "safe"

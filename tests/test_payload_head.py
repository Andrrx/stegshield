import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stegshield.data.synth_lsb import crop_capacity_bytes, embed_sequential_lsb  # noqa: E402
from stegshield.data.torch_dataset import (  # noqa: E402
    ImageRiskDataset,
    payload_bytes_from_target,
    payload_regression_target,
)
from stegshield.labels import STEGO_LABELS  # noqa: E402
from stegshield.models.cnn import StegShieldCNN, create_cnn_model  # noqa: E402
from stegshield.predict_cnn import StegoPredictor  # noqa: E402
from stegshield.train_cnn import _masked_payload_loss  # noqa: E402


def test_multitask_forward_returns_logits_and_payload() -> None:
    model = create_cnn_model("steganalysis", num_classes=2, payload_head=True)
    logits, payload = model.forward_multitask(torch.zeros(3, 3, 64, 64))

    assert tuple(logits.shape) == (3, 2)
    assert tuple(payload.shape) == (3,)
    # forward() still returns logits only, so existing callers are untouched.
    assert tuple(model(torch.zeros(3, 3, 64, 64)).shape) == (3, 2)


def test_detection_only_model_has_no_payload_head() -> None:
    model = create_cnn_model("steganalysis", num_classes=2)
    _, payload = model.forward_multitask(torch.zeros(1, 3, 64, 64))
    assert payload is None


def test_detection_checkpoint_loads_into_detection_model() -> None:
    # A detection-only checkpoint must still load (old-checkpoint compatibility).
    source = StegShieldCNN(num_classes=2)
    target = StegShieldCNN(num_classes=2)
    target.load_state_dict(source.state_dict())


def test_payload_head_rejected_on_yedroudj() -> None:
    with pytest.raises(ValueError, match="payload regression head"):
        create_cnn_model("yedroudj", num_classes=2, payload_head=True)


def test_payload_target_roundtrip() -> None:
    capacity = crop_capacity_bytes(256)
    for payload in [0, 16, 480, 24576]:
        target = payload_regression_target(payload, capacity)
        assert payload_bytes_from_target(target, capacity) == payload
    # Targets are capped at capacity.
    assert payload_regression_target(10**9, capacity) == math.log2(capacity + 1)


def test_masked_payload_loss_ignores_nan_targets() -> None:
    estimate = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([math.nan, 2.0, math.nan])  # only index 1 supervised

    loss, abs_error, count = _masked_payload_loss(estimate, targets, capacity_bytes=24576)

    assert count == 1
    assert abs_error == 0.0  # estimate matches target exactly on the supervised sample
    assert float(loss) == 0.0


def test_masked_payload_loss_all_masked_is_zero_without_nan() -> None:
    estimate = torch.tensor([1.0, 2.0])
    targets = torch.tensor([math.nan, math.nan])

    loss, abs_error, count = _masked_payload_loss(estimate, targets, capacity_bytes=24576)

    assert count == 0
    assert abs_error == 0.0
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_dataset_payload_target_masks_clean_and_keeps_stego(tmp_path: Path) -> None:
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    Image.fromarray(base, mode="RGB").save(tmp_path / "clean.png")
    Image.fromarray(embed_sequential_lsb(base, b"\x01" * 480), mode="RGB").save(
        tmp_path / "stego.png"
    )
    csv = tmp_path / "regress.csv"
    csv.write_text(
        "path,label,payload_bytes\n"
        f"{(tmp_path / 'clean.png')},safe,\n"
        f"{(tmp_path / 'stego.png')},dangerous,480\n",
        encoding="utf-8",
    )

    dataset = ImageRiskDataset(csv, image_size=64, with_payload_target=True)
    _, clean_label, clean_target = dataset[0]
    _, stego_label, stego_target = dataset[1]

    assert clean_label == 0 and math.isnan(clean_target)
    assert stego_label == 1
    assert stego_target == pytest.approx(payload_regression_target(480, crop_capacity_bytes(64)))


def _save_payload_checkpoint(tmp_path: Path, image_size: int = 64) -> Path:
    model = create_cnn_model("steganalysis", num_classes=2, payload_head=True)
    path = tmp_path / "multitask.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": image_size,
            "model_name": "steganalysis",
            "normalization": "raw255",
            "crop": "top-left",
            "task": "stego",
            "payload_head": True,
            "payload_loss_weight": 0.5,
            "payload_capacity_bytes": crop_capacity_bytes(image_size),
        },
        path,
    )
    return path


def test_predict_with_payload_returns_byte_estimate(tmp_path: Path) -> None:
    model_path = _save_payload_checkpoint(tmp_path)
    image_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    predictor = StegoPredictor(model_path=model_path, device="cpu")
    probability, payload_bytes = predictor.predict_with_payload(image_path)

    assert 0.0 <= probability <= 1.0
    assert payload_bytes is not None
    assert 0 <= payload_bytes <= crop_capacity_bytes(64)


def test_predict_with_payload_none_for_detection_only_checkpoint(tmp_path: Path) -> None:
    model = create_cnn_model("steganalysis", num_classes=2)
    model_path = tmp_path / "detect.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": 64,
            "model_name": "steganalysis",
            "normalization": "raw255",
            "crop": "top-left",
            "task": "stego",
        },
        model_path,
    )
    image_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    predictor = StegoPredictor(model_path=model_path, device="cpu")
    _, payload_bytes = predictor.predict_with_payload(image_path)
    assert payload_bytes is None

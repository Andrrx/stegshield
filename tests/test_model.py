import pytest

torch = pytest.importorskip("torch")

from stegshield.models.cnn import StegShieldCNN  # noqa: E402


def test_custom_cnn_output_shape() -> None:
    model = StegShieldCNN(num_classes=3)
    output = model(torch.zeros(2, 3, 224, 224))

    assert tuple(output.shape) == (2, 3)

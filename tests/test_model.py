import pytest

torch = pytest.importorskip("torch")

from stegshield.models.cnn import SteganalysisPreprocessor, StegShieldCNN  # noqa: E402


def test_custom_cnn_output_shape() -> None:
    model = StegShieldCNN(num_classes=3)
    output = model(torch.zeros(2, 3, 224, 224))

    assert tuple(output.shape) == (2, 3)


def test_preprocessor_applies_rgb_high_pass_filter_bank() -> None:
    preprocessor = SteganalysisPreprocessor(input_channels=3)
    output = preprocessor(torch.zeros(2, 3, 32, 32))

    assert tuple(output.shape) == (2, 15, 32, 32)
    assert preprocessor.high_pass.groups == 3
    assert preprocessor.high_pass.weight.requires_grad is False

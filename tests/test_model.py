import pytest

torch = pytest.importorskip("torch")

from stegshield.models.cnn import (  # noqa: E402
    SRMPreprocessor,
    SteganalysisPreprocessor,
    StegShieldCNN,
    TruncationLinearUnit,
    YedroudjNet,
    create_cnn_model,
    srm_filter_bank,
)


def test_custom_cnn_output_shape() -> None:
    model = StegShieldCNN(num_classes=2)
    output = model(torch.zeros(2, 3, 224, 224))

    assert tuple(output.shape) == (2, 2)


def test_yedroudj_net_output_shape() -> None:
    model = YedroudjNet(num_classes=2)
    output = model(torch.zeros(2, 3, 256, 256))

    assert tuple(output.shape) == (2, 2)


def test_create_cnn_model_selects_model_variant() -> None:
    assert isinstance(create_cnn_model("yedroudj", num_classes=2), YedroudjNet)
    assert isinstance(create_cnn_model("steganalysis", num_classes=2), StegShieldCNN)


def test_create_cnn_model_rejects_removed_baseline() -> None:
    with pytest.raises(ValueError, match="Yedroudj-Net"):
        create_cnn_model("baseline", num_classes=2)


def test_srm_filter_bank_has_30_unnormalized_kernels() -> None:
    kernels = srm_filter_bank()

    assert tuple(kernels.shape) == (30, 1, 5, 5)
    # SRM high-pass kernels must each sum to zero (flat regions produce no residual).
    assert torch.allclose(kernels.sum(dim=(2, 3)), torch.zeros(30, 1), atol=1e-6)
    # Unnormalized values as in Yedroudj-Net: the 5x5 SQUARE kernel keeps -12 at center.
    assert kernels.abs().max().item() == 12.0


def test_srm_preprocessor_applies_filters_per_channel() -> None:
    preprocessor = SRMPreprocessor(in_channels=3)
    output = preprocessor(torch.zeros(2, 3, 32, 32))

    assert tuple(output.shape) == (2, 90, 32, 32)
    assert preprocessor.high_pass.groups == 3
    assert preprocessor.high_pass.weight.requires_grad is False


def test_steganalysis_preprocessor_truncates_srm_residuals() -> None:
    preprocessor = SteganalysisPreprocessor(input_channels=3)
    output = preprocessor(torch.rand(2, 3, 32, 32) * 255.0)

    assert tuple(output.shape) == (2, 90, 32, 32)
    assert preprocessor.high_pass.weight.requires_grad is False
    assert output.max().item() <= 3.0
    assert output.min().item() >= -3.0


def test_truncation_linear_unit_clips_residual_values() -> None:
    truncation = TruncationLinearUnit(threshold=2.0)
    output = truncation(torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0]))

    assert output.tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]

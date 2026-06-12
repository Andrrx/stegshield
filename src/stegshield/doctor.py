from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class TorchDoctorReport:
    python_version: str
    torch_version: str
    cuda_available: bool
    torch_cuda_version: str | None
    cudnn_enabled: bool
    cuda_device_count: int
    cuda_device_names: list[str]
    current_cuda_device: int | None
    current_cuda_device_name: str | None
    requested_device: str
    requested_device_valid: bool
    warnings: list[str]


def build_torch_doctor_report(requested_device: str = "cuda") -> TorchDoctorReport:
    import torch

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_names = [
        torch.cuda.get_device_name(index) for index in range(cuda_device_count)
    ]
    current_cuda_device = torch.cuda.current_device() if cuda_available else None
    current_cuda_device_name = (
        torch.cuda.get_device_name(current_cuda_device)
        if current_cuda_device is not None
        else None
    )
    requested_device_valid = _requested_device_valid(requested_device, cuda_device_count)
    warnings = _warnings(requested_device, cuda_available, requested_device_valid)

    return TorchDoctorReport(
        python_version=f"{platform.python_implementation()} {sys.version.split()[0]}",
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        torch_cuda_version=torch.version.cuda,
        cudnn_enabled=torch.backends.cudnn.enabled,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_device_names,
        current_cuda_device=current_cuda_device,
        current_cuda_device_name=current_cuda_device_name,
        requested_device=requested_device,
        requested_device_valid=requested_device_valid,
        warnings=warnings,
    )


def training_device_info(requested_device: str, resolved_device: object) -> dict[str, object]:
    import torch

    cuda_available = torch.cuda.is_available()
    device_name = None
    if getattr(resolved_device, "type", None) == "cuda" and cuda_available:
        device_index = resolved_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)

    return {
        "requested_device": requested_device,
        "resolved_torch_device": str(resolved_device),
        "cuda_available": cuda_available,
        "selected_cuda_device_name": device_name,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_enabled": torch.backends.cudnn.enabled,
    }


def _requested_device_valid(requested_device: str, cuda_device_count: int) -> bool:
    if requested_device == "cpu":
        return True
    if requested_device == "cuda":
        return cuda_device_count > 0
    if requested_device.startswith("cuda:"):
        try:
            index = int(requested_device.split(":", maxsplit=1)[1])
        except ValueError:
            return False
        return 0 <= index < cuda_device_count
    return False


def _warnings(
    requested_device: str,
    cuda_available: bool,
    requested_device_valid: bool,
) -> list[str]:
    warnings = []
    if not cuda_available:
        warnings.append("CUDA is not available to PyTorch. Training will run on CPU.")
    if requested_device == "cpu":
        warnings.append("Requested device is CPU. Use --device cuda for NVIDIA GPU training.")
    elif not requested_device_valid:
        warnings.append(f"Requested device is not valid or unavailable: {requested_device}")
    return warnings

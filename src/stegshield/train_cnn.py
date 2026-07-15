from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from stegshield.data.synth_lsb import crop_capacity_bytes
from stegshield.data.torch_dataset import ImageRiskDataset
from stegshield.doctor import training_device_info
from stegshield.labels import label_to_index_for_task, labels_for_task
from stegshield.ml_metrics import (
    better_selection_metric,
    classification_report_from_confusion,
    empty_confusion,
    update_confusion,
)
from stegshield.models.cnn import create_cnn_model


@dataclass(frozen=True)
class TrainingConfig:
    train_csv: Path
    val_csv: Path
    output_model: Path
    output_metrics: Path
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    image_size: int = 256
    device: str = "cpu"
    model_name: str = "steganalysis"
    normalization: str = "raw255"
    crop: str = "top-left"
    task: str = "stego"
    class_weights: bool = True
    balanced_sampler: bool = True
    selection_metric: str = "macro_f1"
    num_workers: int = 0
    amp: bool = False
    payload_head: bool = False
    payload_loss_weight: float = 0.5
    augment: bool = False


def train_cnn(config: TrainingConfig) -> dict[str, object]:
    started_at = datetime.now(UTC).isoformat()
    if config.selection_metric not in {"macro_f1", "balanced_accuracy"}:
        raise ValueError("selection_metric must be macro_f1 or balanced_accuracy.")

    if config.payload_head and config.model_name != "steganalysis":
        raise ValueError("The payload regression head is only available on the steganalysis model.")

    device = torch.device(config.device)
    train_dataset = ImageRiskDataset(
        config.train_csv,
        image_size=config.image_size,
        normalization=config.normalization,
        task=config.task,
        crop=config.crop,
        with_payload_target=config.payload_head,
        augment=config.augment,
    )
    # Validation is never augmented: it measures pristine-image performance.
    val_dataset = ImageRiskDataset(
        config.val_csv,
        image_size=config.image_size,
        normalization=config.normalization,
        task=config.task,
        crop=config.crop,
        with_payload_target=config.payload_head,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training split is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")

    pin_memory = device.type == "cuda"
    persistent_workers = config.num_workers > 0
    train_sampler = (
        _balanced_sampler(train_dataset.samples, config.task)
        if config.balanced_sampler
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    labels = labels_for_task(config.task)
    model = create_cnn_model(
        model_name=config.model_name,
        num_classes=len(labels),
        payload_head=config.payload_head,
    ).to(device)
    payload_loss_weight = config.payload_loss_weight if config.payload_head else None
    capacity_bytes = crop_capacity_bytes(config.image_size)
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_dataset.samples, config.task, device)
        if config.class_weights
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    device_info = training_device_info(config.device, device)
    device_info["pin_memory"] = pin_memory
    device_info["num_workers"] = config.num_workers
    device_info["amp"] = use_amp
    history: list[dict[str, object]] = []
    best_selection_metric: float | None = None
    best_epoch: int | None = None
    best_val_metrics: dict[str, object] | None = None

    print(
        f"Training {config.model_name} on {len(train_dataset)} train / "
        f"{len(val_dataset)} val samples for {config.epochs} epochs "
        f"(device={config.device}, batch_size={config.batch_size}, amp={use_amp})",
        flush=True,
    )

    training_started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.monotonic()
        train_loss, train_report = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            labels=labels,
            use_amp=use_amp,
            scaler=scaler,
            payload_loss_weight=payload_loss_weight,
            capacity_bytes=capacity_bytes,
        )
        val_loss, val_report = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            labels=labels,
            use_amp=use_amp,
            scaler=scaler,
            payload_loss_weight=payload_loss_weight,
            capacity_bytes=capacity_bytes,
        )
        scheduler.step()

        selection_value = float(val_report[config.selection_metric])
        epoch_seconds = time.monotonic() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metrics": train_report,
                "val_loss": val_loss,
                "val_metrics": val_report,
                "selection_metric_name": config.selection_metric,
                "selection_metric_value": selection_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_seconds": round(epoch_seconds, 1),
            }
        )

        improved = better_selection_metric(
            candidate=selection_value,
            current_best=best_selection_metric,
            candidate_epoch=epoch,
            best_epoch=best_epoch,
        )
        if improved:
            best_selection_metric = selection_value
            best_epoch = epoch
            best_val_metrics = val_report
            _save_checkpoint(
                model=model,
                config=config,
                labels=labels,
                device_info=device_info,
                started_at=started_at,
                best_epoch=best_epoch,
                best_selection_metric=best_selection_metric,
            )

        average_epoch = (time.monotonic() - training_started) / epoch
        remaining_minutes = average_epoch * (config.epochs - epoch) / 60
        print(
            f"Epoch {epoch}/{config.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_{config.selection_metric}={selection_value:.4f} "
            f"val_balanced_accuracy={float(val_report['balanced_accuracy']):.4f} "
            f"[{epoch_seconds / 60:.1f} min/epoch, ~{remaining_minutes:.0f} min left]"
            f"{' *best, checkpoint saved*' if improved else ''}",
            flush=True,
        )

        _write_metrics(
            config=config,
            device_info=device_info,
            started_at=started_at,
            history=history,
            best_epoch=best_epoch,
            best_selection_metric=best_selection_metric,
            best_val_metrics=best_val_metrics,
        )

    return _write_metrics(
        config=config,
        device_info=device_info,
        started_at=started_at,
        history=history,
        best_epoch=best_epoch,
        best_selection_metric=best_selection_metric,
        best_val_metrics=best_val_metrics,
    )


def _checkpoint_metadata(
    config: TrainingConfig,
    labels: tuple[str, ...],
    device_info: dict[str, object],
    started_at: str,
    best_epoch: int | None,
    best_selection_metric: float | None,
) -> dict[str, object]:
    return {
        "task": config.task,
        "labels": labels,
        "model_name": config.model_name,
        "normalization": config.normalization,
        "crop": config.crop,
        "class_weights": config.class_weights,
        "balanced_sampler": config.balanced_sampler,
        "selection_metric": config.selection_metric,
        "best_epoch": best_epoch,
        "best_selection_metric": best_selection_metric,
        "best_selection_metric_name": config.selection_metric,
        "train_csv": str(config.train_csv),
        "val_csv": str(config.val_csv),
        "output_model": str(config.output_model),
        "output_metrics": str(config.output_metrics),
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "device": config.device,
        "num_workers": config.num_workers,
        "amp": config.amp,
        "augment": config.augment,
        "payload_head": config.payload_head,
        "payload_loss_weight": config.payload_loss_weight if config.payload_head else None,
        "payload_capacity_bytes": crop_capacity_bytes(config.image_size)
        if config.payload_head
        else None,
        "timestamp": started_at,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "device_info": device_info,
    }


def _save_checkpoint(
    model: nn.Module,
    config: TrainingConfig,
    labels: tuple[str, ...],
    device_info: dict[str, object],
    started_at: str,
    best_epoch: int | None,
    best_selection_metric: float | None,
) -> None:
    """Persist the current best model immediately so interrupted runs keep it."""
    config.output_model.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": state_dict,
            **_checkpoint_metadata(
                config=config,
                labels=labels,
                device_info=device_info,
                started_at=started_at,
                best_epoch=best_epoch,
                best_selection_metric=best_selection_metric,
            ),
        },
        config.output_model,
    )


def _write_metrics(
    config: TrainingConfig,
    device_info: dict[str, object],
    started_at: str,
    history: list[dict[str, object]],
    best_epoch: int | None,
    best_selection_metric: float | None,
    best_val_metrics: dict[str, object] | None,
) -> dict[str, object]:
    """Write metrics after every epoch so progress survives interruption."""
    metrics = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "metadata": _checkpoint_metadata(
            config=config,
            labels=labels_for_task(config.task),
            device_info=device_info,
            started_at=started_at,
            best_epoch=best_epoch,
            best_selection_metric=best_selection_metric,
        ),
        "best_epoch": best_epoch,
        "best_selection_metric": best_selection_metric,
        "best_selection_metric_name": config.selection_metric,
        "best_val_metrics": best_val_metrics,
        "best_val_accuracy": _best_history_metric(history, "accuracy"),
        "device_info": device_info,
        "history": history,
    }
    config.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    config.output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    labels: tuple[str, ...],
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    payload_loss_weight: float | None = None,
    capacity_bytes: int | None = None,
) -> tuple[float, dict[str, object]]:
    is_training = optimizer is not None
    model.train(is_training)
    multitask = payload_loss_weight is not None

    total_loss = 0.0
    total = 0
    confusion = empty_confusion(len(labels))
    payload_abs_error_sum = 0.0
    payload_supervised_count = 0

    with torch.set_grad_enabled(is_training):
        for batch in loader:
            if multitask:
                images, batch_labels, payload_targets = batch
                payload_targets = payload_targets.to(device, non_blocking=True).float()
            else:
                images, batch_labels = batch
                payload_targets = None

            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                if multitask:
                    logits, payload_estimate = model.forward_multitask(images)
                else:
                    logits = model(images)
                    payload_estimate = None
                loss = criterion(logits, batch_labels)
                if multitask:
                    payload_loss, batch_abs_error, batch_count = _masked_payload_loss(
                        payload_estimate, payload_targets, capacity_bytes
                    )
                    loss = loss + payload_loss_weight * payload_loss
                    payload_abs_error_sum += batch_abs_error
                    payload_supervised_count += batch_count

            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            predictions = logits.argmax(dim=1)
            for actual, predicted in zip(batch_labels.cpu(), predictions.cpu(), strict=True):
                update_confusion(confusion, int(actual), int(predicted))

    report = classification_report_from_confusion(confusion, labels)
    if multitask:
        report["payload_mae_bytes"] = (
            round(payload_abs_error_sum / payload_supervised_count, 3)
            if payload_supervised_count > 0
            else None
        )
        report["payload_supervised_count"] = payload_supervised_count
    return total_loss / total, report


def _masked_payload_loss(
    payload_estimate: torch.Tensor,
    payload_targets: torch.Tensor,
    capacity_bytes: int | None,
) -> tuple[torch.Tensor, float, int]:
    """Smooth-L1 over supervised samples only; NaN targets (clean/unknown) are dropped.

    Returns (loss, summed absolute byte error, supervised sample count). The loss
    is a zero tensor wired to the estimate when the batch has no supervision, so
    the graph stays valid without contributing gradient.
    """
    mask = ~torch.isnan(payload_targets)
    if not bool(mask.any()):
        return payload_estimate.sum() * 0.0, 0.0, 0

    estimate = payload_estimate[mask].float()
    target = payload_targets[mask].float()
    loss = nn.functional.smooth_l1_loss(estimate, target)

    cap = capacity_bytes if capacity_bytes is not None else 0
    estimate_bytes = torch.clamp(2.0 ** estimate.detach() - 1.0, min=0.0, max=float(cap))
    target_bytes = torch.clamp(2.0**target - 1.0, min=0.0, max=float(cap))
    abs_error = float((estimate_bytes - target_bytes).abs().sum().item())
    return loss, abs_error, int(mask.sum().item())


def _class_weights(samples: list[object], task: str, device: torch.device) -> torch.Tensor:
    labels = labels_for_task(task)
    counts = torch.zeros(len(labels), dtype=torch.float32)
    for sample in samples:
        counts[label_to_index_for_task(sample.label, task)] += 1

    weights = counts.sum() / (len(labels) * counts.clamp_min(1))
    return weights.to(device)


def _balanced_sampler(samples: list[object], task: str) -> WeightedRandomSampler:
    labels = labels_for_task(task)
    counts = torch.zeros(len(labels), dtype=torch.float32)
    sample_labels = []
    for sample in samples:
        label_index = label_to_index_for_task(sample.label, task)
        sample_labels.append(label_index)
        counts[label_index] += 1

    weights = [float(1.0 / counts[label_index].clamp_min(1).item()) for label_index in sample_labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _best_history_metric(history: list[dict[str, object]], metric_name: str) -> float | None:
    values = []
    for epoch in history:
        val_metrics = epoch["val_metrics"]
        if isinstance(val_metrics, dict):
            values.append(float(val_metrics[metric_name]))
    return max(values, default=None)

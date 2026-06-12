from __future__ import annotations

LABELS: tuple[str, ...] = ("safe", "suspicious", "dangerous")
LABEL_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_INDEX.items()}

STEGO_LABELS: tuple[str, ...] = ("clean", "stego")
STEGO_LABEL_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(STEGO_LABELS)}
STEGO_INDEX_TO_LABEL: dict[int, str] = {
    index: label for label, index in STEGO_LABEL_TO_INDEX.items()
}


def labels_for_task(task: str) -> tuple[str, ...]:
    if task == "risk":
        return LABELS
    if task == "stego":
        return STEGO_LABELS
    raise ValueError(f"Unsupported CNN task: {task}")


def label_to_index_for_task(label: str, task: str) -> int:
    if task == "risk":
        return LABEL_TO_INDEX[label]
    if task == "stego":
        return STEGO_LABEL_TO_INDEX["clean" if label == "safe" else "stego"]
    raise ValueError(f"Unsupported CNN task: {task}")

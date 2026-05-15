from __future__ import annotations

LABELS: tuple[str, ...] = ("safe", "suspicious", "dangerous")
LABEL_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_INDEX.items()}

"""Project configuration helpers and default tabletop classes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import read_yaml


DEFAULT_TABLETOP_CLASSES: list[str] = [
    "bottle",
    "cup",
    "bowl",
    "fork",
    "knife",
    "spoon",
    "apple",
    "banana",
    "book",
    "cell phone",
]

BACKGROUND_CLASS = "__background__"


@dataclass(frozen=True)
class DatasetConfig:
    """Paths and class names for a COCO-format detection dataset."""

    image_dir: Path
    annotation_path: Path
    classes: list[str]


def with_background(classes: list[str]) -> list[str]:
    """Return class names in TorchVision detection order."""

    return [BACKGROUND_CLASS, *classes]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    return read_yaml(path)


def resolve_classes(classes: list[str] | None = None) -> list[str]:
    """Return user-supplied classes or the default tabletop class list."""

    return list(classes) if classes else list(DEFAULT_TABLETOP_CLASSES)

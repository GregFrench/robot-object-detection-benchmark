"""Small utility functions shared by training and evaluation scripts."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path) -> Any:
    """Load a JSON file with a clear error for missing paths."""

    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> Path:
    """Write JSON with stable indentation."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    return output_path


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file using PyYAML."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read config files. Install with `pip install PyYAML`.") from exc

    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a YAML file using PyYAML."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write dataset YAML files. Install with `pip install PyYAML`.") from exc

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return output_path


def save_csv(rows: Iterable[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    """Write dictionaries to CSV."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError("Cannot infer CSV columns from an empty row list.")
    columns = fieldnames or list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def validate_file(path: str | Path, label: str) -> Path:
    """Return a file path or raise a helpful FileNotFoundError."""

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


def validate_dir(path: str | Path, label: str) -> Path:
    """Return a directory path or raise a helpful FileNotFoundError."""

    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


def set_seed(seed: int) -> None:
    """Seed common random number generators for reproducible demos."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preferred: str = "auto"):
    """Return a torch.device from a user-facing device string."""

    import torch

    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use `--device cpu` for a CPU smoke test.")
    return device


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print a compact fixed-width table without extra dependencies."""

    if not rows:
        print("No rows to display.")
        return
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in rows:
        print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))

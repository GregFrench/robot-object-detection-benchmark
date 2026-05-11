#!/usr/bin/env python3
"""Compare metric JSON files from Faster R-CNN and YOLO experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a comparison table from model metrics JSON files.")
    parser.add_argument("--metrics", nargs="*", type=Path, default=None, help="Specific metric JSON files to compare.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"), help="Directory to scan for JSON metrics.")
    parser.add_argument("--output", type=Path, default=Path("outputs/model_comparison.csv"), help="Comparison CSV path.")
    return parser.parse_args()


def metric_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def load_metric_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "model_name": metric_value(data, "model_name", "name") or path.stem,
        "mAP_or_simplified_mAP": metric_value(data, "simplified_map", "mAP", "map", "map50"),
        "precision": metric_value(data, "precision", "precision_box"),
        "recall": metric_value(data, "recall", "recall_box"),
        "average_inference_latency_ms": metric_value(data, "mean_latency_ms", "avg_latency_ms", "latency_ms"),
        "notes": metric_value(data, "metric_type", "notes") or "user-provided metrics",
        "source_file": str(path),
    }


def main() -> int:
    args = parse_args()
    import pandas as pd

    metric_files = args.metrics or sorted(args.metrics_dir.glob("*.json"))
    if not metric_files:
        raise FileNotFoundError("No metric JSON files found. Pass --metrics or populate --metrics-dir.")

    rows = [load_metric_file(path) for path in metric_files]
    dataframe = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False)
    print(dataframe.to_string(index=False))
    print(f"\nSaved comparison CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

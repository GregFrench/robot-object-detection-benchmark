#!/usr/bin/env python3
"""Train an Ultralytics YOLO model on a prepared dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with the optional Ultralytics integration.")
    parser.add_argument("--data-yaml", type=Path, required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO weights/model name, for example yolo11n.pt.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, for example 0, cpu, or None.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolo_runs"), help="YOLO run output directory.")
    parser.add_argument("--run-name", default="tabletop_yolo", help="YOLO run name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_yaml.is_file():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {args.data_yaml}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Ultralytics is not installed. Install it with `pip install ultralytics` "
            "or uncomment the optional dependency in requirements.txt."
        )
        return 1

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "project": str(args.output_dir),
        "name": args.run_name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)
    print(f"YOLO training complete. Results saved under {args.output_dir / args.run_name}")
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

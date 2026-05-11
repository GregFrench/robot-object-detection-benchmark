#!/usr/bin/env python3
"""Train TorchVision Faster R-CNN on a COCO-format dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for tabletop object detection.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Image directory for a COCO-format dataset.")
    parser.add_argument("--annotations", type=Path, default=None, help="COCO annotation JSON.")
    parser.add_argument("--num-classes", type=int, default=None, help="Classes including background. Inferred if omitted.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.005, help="SGD learning rate.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for checkpoints and logs.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--mock-data", action="store_true", help="Use a tiny synthetic dataset for smoke tests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src.train_faster_rcnn import FasterRCNNTrainingConfig, run_training

    config = FasterRCNNTrainingConfig(
        data_dir=args.data_dir,
        annotations=args.annotations,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        mock_data=args.mock_data,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    summary = run_training(config)
    print("Training complete.")
    print(f"Latest checkpoint: {summary['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a Faster R-CNN checkpoint on a COCO-format validation set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN with simplified IoU-based metrics.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Faster R-CNN checkpoint path.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Validation image directory.")
    parser.add_argument("--annotations", type=Path, default=None, help="Validation COCO annotation JSON.")
    parser.add_argument("--num-classes", type=int, default=None, help="Classes including background. Inferred if omitted.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"), help="Metrics output directory.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Prediction score threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for a true positive.")
    parser.add_argument("--model-name", default="faster_rcnn", help="Name stored in metrics files.")
    parser.add_argument("--mock-data", action="store_true", help="Use synthetic data and an untrained model for smoke tests.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src.config import with_background
    from src.datasets import CocoDetectionDataset, MockDetectionDataset, load_coco_class_names
    from src.evaluate import run_faster_rcnn_evaluation, save_evaluation_result
    from src.models import build_faster_rcnn, load_faster_rcnn_checkpoint
    from src.transforms import get_eval_transform
    from src.utils import get_device

    device = get_device(args.device)

    if args.mock_data:
        foreground_classes = max((args.num_classes or 4) - 1, 1)
        dataset = MockDetectionDataset(num_images=args.max_samples or 4, num_classes=foreground_classes)
        num_classes = len(dataset.class_names) + 1
        model = build_faster_rcnn(num_classes=num_classes, pretrained=False)
        model.to(device)
    else:
        if args.data_dir is None or args.annotations is None or args.checkpoint is None:
            raise ValueError("`--checkpoint`, `--data-dir`, and `--annotations` are required unless `--mock-data` is used.")
        class_names = load_coco_class_names(args.annotations)
        dataset = CocoDetectionDataset(
            image_dir=args.data_dir,
            annotation_path=args.annotations,
            classes=class_names,
            transforms=get_eval_transform(),
            max_samples=args.max_samples,
        )
        num_classes = args.num_classes or len(with_background(class_names))
        model, _checkpoint = load_faster_rcnn_checkpoint(args.checkpoint, num_classes=num_classes, device=device)

    result = run_faster_rcnn_evaluation(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        model_name=args.model_name,
    )
    paths = save_evaluation_result(result, args.output_dir, args.model_name)
    print(f"Saved metrics JSON: {paths['json']}")
    print(f"Saved metrics CSV: {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

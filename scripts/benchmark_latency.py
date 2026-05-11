#!/usr/bin/env python3
"""Benchmark Faster R-CNN inference latency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure average Faster R-CNN inference latency over N images.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Faster R-CNN checkpoint path.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Image directory.")
    parser.add_argument("--annotations", type=Path, default=None, help="COCO annotation JSON.")
    parser.add_argument("--num-classes", type=int, default=None, help="Classes including background. Inferred if omitted.")
    parser.add_argument("--num-images", type=int, default=32, help="Number of timed images.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup inference iterations.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latency"), help="Latency result directory.")
    parser.add_argument("--model-name", default="faster_rcnn", help="Model name stored in JSON.")
    parser.add_argument("--mock-data", action="store_true", help="Use synthetic images and an untrained model for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src.config import with_background
    from src.datasets import CocoDetectionDataset, MockDetectionDataset, load_coco_class_names
    from src.latency import benchmark_latency
    from src.models import build_faster_rcnn, load_faster_rcnn_checkpoint
    from src.transforms import get_eval_transform
    from src.utils import ensure_dir, get_device, save_json

    device = get_device(args.device)

    if args.mock_data:
        foreground_classes = max((args.num_classes or 4) - 1, 1)
        dataset = MockDetectionDataset(num_images=max(args.num_images, 1), num_classes=foreground_classes)
        class_names = with_background(dataset.class_names)
        model = build_faster_rcnn(num_classes=len(class_names), pretrained=False)
        model.to(device)
    else:
        if args.checkpoint is None or args.data_dir is None or args.annotations is None:
            raise ValueError("`--checkpoint`, `--data-dir`, and `--annotations` are required unless `--mock-data` is used.")
        annotation_names = load_coco_class_names(args.annotations)
        model, checkpoint = load_faster_rcnn_checkpoint(args.checkpoint, num_classes=args.num_classes, device=device)
        checkpoint_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
        foreground_names = checkpoint_names or annotation_names
        missing = sorted(set(foreground_names) - set(annotation_names))
        if missing:
            raise ValueError(f"Checkpoint classes are missing from latency annotations: {missing}")
        if checkpoint_names and foreground_names != annotation_names:
            print("Using checkpoint class order for latency dataset labels.")
        dataset = CocoDetectionDataset(
            image_dir=args.data_dir,
            annotation_path=args.annotations,
            classes=foreground_names,
            transforms=get_eval_transform(),
            max_samples=args.num_images,
        )
        class_names = with_background(foreground_names)

    images = [dataset[index][0] for index in range(min(args.num_images, len(dataset)))]
    metrics = benchmark_latency(model, images, device=device, warmup=args.warmup, iterations=args.num_images)
    metrics["model_name"] = args.model_name
    metrics["num_classes"] = len(class_names)

    output_dir = ensure_dir(args.output_dir)
    output_path = save_json(metrics, output_dir / f"{args.model_name}_latency.json")
    print(f"Mean latency: {metrics['mean_latency_ms']} ms")
    print(f"Median latency: {metrics['median_latency_ms']} ms")
    print(f"Images/sec: {metrics['images_per_second']}")
    print(f"Saved latency JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

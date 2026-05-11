#!/usr/bin/env python3
"""Visualize Faster R-CNN predictions or saved prediction JSON records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw predicted bounding boxes on sample images.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Faster R-CNN checkpoint path.")
    parser.add_argument("--predictions", type=Path, default=None, help="Prediction JSON file to visualize instead of running a model.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Image directory for checkpoint inference.")
    parser.add_argument("--annotations", type=Path, default=None, help="COCO annotations for checkpoint inference.")
    parser.add_argument("--num-classes", type=int, default=None, help="Classes including background. Inferred if omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sample_predictions"), help="Visualization directory.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score to draw.")
    parser.add_argument("--max-images", type=int, default=8, help="Maximum images to visualize.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--mock-data", action="store_true", help="Use synthetic data and an untrained model for smoke tests.")
    return parser.parse_args()


def visualize_prediction_file(args) -> int:
    from PIL import Image

    from src.config import with_background
    from src.utils import ensure_dir
    from src.visualize import draw_predictions

    with args.predictions.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("predictions", payload if isinstance(payload, list) else [])
    class_names = payload.get("class_names", with_background([])) if isinstance(payload, dict) else with_background([])
    output_dir = ensure_dir(args.output_dir)

    for index, record in enumerate(records[: args.max_images]):
        image_path = Path(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        output_path = output_dir / f"prediction_{index:03d}.jpg"
        draw_predictions(image, record, class_names, output_path, args.confidence_threshold, title=image_path.name)
    print(f"Saved visualizations to {output_dir}")
    return 0


def visualize_checkpoint(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    from src.config import with_background
    from src.datasets import CocoDetectionDataset, MockDetectionDataset, collate_fn, load_coco_class_names
    from src.models import build_faster_rcnn, load_faster_rcnn_checkpoint
    from src.transforms import get_eval_transform
    from src.utils import ensure_dir, get_device
    from src.visualize import draw_predictions

    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    if args.mock_data:
        foreground_classes = max((args.num_classes or 4) - 1, 1)
        dataset = MockDetectionDataset(num_images=args.max_images, num_classes=foreground_classes)
        class_names = with_background(dataset.class_names)
        model = build_faster_rcnn(num_classes=len(class_names), pretrained=False)
        model.to(device)
    else:
        if args.checkpoint is None or args.data_dir is None or args.annotations is None:
            raise ValueError("`--checkpoint`, `--data-dir`, and `--annotations` are required unless --predictions or --mock-data is used.")
        foreground_names = load_coco_class_names(args.annotations)
        dataset = CocoDetectionDataset(
            image_dir=args.data_dir,
            annotation_path=args.annotations,
            classes=foreground_names,
            transforms=get_eval_transform(),
            max_samples=args.max_images,
        )
        class_names = with_background(foreground_names)
        num_classes = args.num_classes or len(class_names)
        model, _checkpoint = load_faster_rcnn_checkpoint(args.checkpoint, num_classes=num_classes, device=device)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model.eval()
    saved = 0
    with torch.no_grad():
        for index, (images, _targets) in enumerate(dataloader):
            if index >= args.max_images:
                break
            image = images[0]
            prediction = model([image.to(device)])[0]
            output_path = output_dir / f"prediction_{index:03d}.jpg"
            draw_predictions(image, prediction, class_names, output_path, args.confidence_threshold)
            saved += 1
    print(f"Saved {saved} visualizations to {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.predictions is not None:
        if not args.predictions.is_file():
            raise FileNotFoundError(f"Prediction JSON not found: {args.predictions}")
        return visualize_prediction_file(args)
    return visualize_checkpoint(args)


if __name__ == "__main__":
    raise SystemExit(main())

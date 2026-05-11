#!/usr/bin/env python3
"""Prepare a small COCO subset for tabletop object detection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter local COCO data to tabletop/household object classes.")
    parser.add_argument("--coco-images", type=Path, required=True, help="Directory containing COCO images.")
    parser.add_argument("--coco-annotations", type=Path, required=True, help="Path to COCO instances JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for the filtered subset.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class names to keep. Defaults to the project tabletop class list.",
    )
    parser.add_argument("--split-name", default="train", help="Split name to write, for example train or val.")
    parser.add_argument("--limit-images", type=int, default=None, help="Maximum number of matching images to include.")
    parser.add_argument(
        "--link-method",
        choices=["copy", "symlink"],
        default="copy",
        help="Copy images or create symlinks into the subset directory.",
    )
    parser.add_argument("--create-yolo-labels", action="store_true", help="Also write YOLO-format label files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview the subset without writing files.")
    return parser.parse_args()


def valid_bbox(bbox: list[float]) -> bool:
    return len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0


def resolve_image_path(image_dir: Path, file_name: str) -> Path:
    candidate = image_dir / file_name
    if candidate.is_file():
        return candidate
    return image_dir / Path(file_name).name


def copy_or_symlink(source: Path, destination: Path, method: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if method == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        shutil.copy2(source, destination)


def yolo_line(annotation: dict[str, Any], image_info: dict[str, Any], category_to_yolo_id: dict[int, int]) -> str:
    x_min, y_min, width, height = annotation["bbox"]
    image_width = float(image_info["width"])
    image_height = float(image_info["height"])
    x_center = (x_min + width / 2.0) / image_width
    y_center = (y_min + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    class_id = category_to_yolo_id[annotation["category_id"]]
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def main() -> int:
    args = parse_args()

    from src.config import DEFAULT_TABLETOP_CLASSES
    from src.utils import save_json, write_yaml

    if not args.coco_annotations.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {args.coco_annotations}")
    if not args.coco_images.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {args.coco_images}")

    target_classes = args.classes or DEFAULT_TABLETOP_CLASSES
    with args.coco_annotations.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    categories = coco.get("categories", [])
    categories_by_name = {category["name"]: category for category in categories}
    missing = sorted(set(target_classes) - set(categories_by_name))
    if missing:
        raise ValueError(f"Requested classes not found in COCO annotations: {missing}")

    selected_categories = [categories_by_name[name] for name in target_classes]
    selected_category_ids = {category["id"] for category in selected_categories}
    category_to_yolo_id = {category["id"]: index for index, category in enumerate(selected_categories)}

    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    target_annotations = [
        annotation
        for annotation in coco.get("annotations", [])
        if annotation.get("category_id") in selected_category_ids and valid_bbox(annotation.get("bbox", []))
    ]
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in target_annotations:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    selected_image_ids = [
        image_id
        for image_id in sorted(annotations_by_image)
        if image_id in images_by_id and resolve_image_path(args.coco_images, images_by_id[image_id]["file_name"]).is_file()
    ]
    if args.limit_images is not None:
        selected_image_ids = selected_image_ids[: args.limit_images]

    selected_image_id_set = set(selected_image_ids)
    filtered_images = [images_by_id[image_id] for image_id in selected_image_ids]
    filtered_annotations = [
        annotation for annotation in target_annotations if int(annotation["image_id"]) in selected_image_id_set
    ]

    class_counts = Counter(
        categories_by_name[name]["name"]
        for annotation in filtered_annotations
        for name in target_classes
        if categories_by_name[name]["id"] == annotation["category_id"]
    )
    summary = {
        "split": args.split_name,
        "num_images": len(filtered_images),
        "num_annotations": len(filtered_annotations),
        "classes": target_classes,
        "class_counts": dict(class_counts),
        "link_method": args.link_method,
        "yolo_labels": bool(args.create_yolo_labels),
    }

    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("Dry run complete. No files were written.")
        return 0

    annotation_dir = args.output_dir / "annotations"
    image_output_dir = args.output_dir / "images" / args.split_name
    label_output_dir = args.output_dir / "labels" / args.split_name
    annotation_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    for image in filtered_images:
        source = resolve_image_path(args.coco_images, image["file_name"])
        destination = image_output_dir / image["file_name"]
        copy_or_symlink(source, destination, args.link_method)

    filtered_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": selected_categories,
    }
    save_json(filtered_coco, annotation_dir / f"instances_{args.split_name}.json")
    save_json(summary, args.output_dir / f"dataset_summary_{args.split_name}.json")

    if args.create_yolo_labels:
        label_output_dir.mkdir(parents=True, exist_ok=True)
        annotations_for_labels: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in filtered_annotations:
            annotations_for_labels[int(annotation["image_id"])].append(annotation)

        for image in filtered_images:
            relative_label_path = Path(image["file_name"]).with_suffix(".txt")
            label_path = label_output_dir / relative_label_path
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                yolo_line(annotation, image, category_to_yolo_id)
                for annotation in annotations_for_labels.get(int(image["id"]), [])
            ]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        val_path = "images/val" if (args.output_dir / "images" / "val").exists() else f"images/{args.split_name}"
        dataset_yaml = {
            "path": str(args.output_dir),
            "train": "images/train" if (args.output_dir / "images" / "train").exists() else f"images/{args.split_name}",
            "val": val_path,
            "nc": len(target_classes),
            "names": target_classes,
        }
        write_yaml(dataset_yaml, args.output_dir / "yolo_dataset.yaml")

    print(f"Wrote subset to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

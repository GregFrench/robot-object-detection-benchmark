"""COCO-format and mock datasets for object detection experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import DEFAULT_TABLETOP_CLASSES
from .transforms import get_eval_transform
from .utils import load_json


Target = dict[str, torch.Tensor]


def collate_fn(batch: list[tuple[torch.Tensor, Target]]) -> tuple[list[torch.Tensor], list[Target]]:
    """TorchVision detection models expect lists of images and targets."""

    images, targets = zip(*batch)
    return list(images), list(targets)


def _valid_bbox_xywh(bbox: list[float]) -> bool:
    return len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0


def _categories_in_file_order(coco: dict[str, Any]) -> list[dict[str, Any]]:
    """Return categories in JSON order after basic validation."""

    categories = coco.get("categories", [])
    names = [category.get("name") for category in categories]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"COCO categories contain duplicate names: {duplicate_names}")
    return categories


def load_coco_class_names(annotation_path: str | Path, classes: list[str] | None = None) -> list[str]:
    """Load class names from a COCO annotation file, preserving JSON order."""

    coco = load_json(annotation_path)
    categories = _categories_in_file_order(coco)
    available = [category["name"] for category in categories]
    if classes is None:
        return available
    missing = sorted(set(classes) - set(available))
    if missing:
        raise ValueError(f"Classes not present in COCO annotations: {missing}")
    return list(classes)


class CocoDetectionDataset(Dataset):
    """Minimal COCO detection dataset for TorchVision Faster R-CNN.

    Labels are remapped to contiguous IDs starting at 1 because TorchVision
    reserves 0 for the background class.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_path: str | Path,
        classes: list[str] | None = None,
        transforms: Callable[[Image.Image | torch.Tensor, Target], tuple[torch.Tensor, Target]] | None = None,
        include_empty: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_path = Path(annotation_path)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {self.annotation_path}")

        self.coco = load_json(self.annotation_path)
        categories = _categories_in_file_order(self.coco)
        category_by_id = {category["id"]: category for category in categories}

        if classes is None:
            self.class_names = [category["name"] for category in categories]
        else:
            available = {category["name"] for category in categories}
            missing = sorted(set(classes) - available)
            if missing:
                raise ValueError(f"Classes not present in COCO annotations: {missing}")
            self.class_names = list(classes)

        selected_ids = {
            category_id
            for category_id, category in category_by_id.items()
            if category["name"] in set(self.class_names)
        }
        self.category_id_to_label = {
            category_id: self.class_names.index(category_by_id[category_id]["name"]) + 1
            for category_id in selected_ids
        }

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.coco.get("annotations", []):
            if annotation.get("category_id") not in selected_ids:
                continue
            bbox = annotation.get("bbox", [])
            if not _valid_bbox_xywh(bbox):
                continue
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        image_records = []
        for image in self.coco.get("images", []):
            image_id = int(image["id"])
            if not include_empty and not annotations_by_image.get(image_id):
                continue
            image_path = self.resolve_image_path(image["file_name"])
            if not image_path.is_file():
                continue
            image_records.append(image)

        image_records = sorted(image_records, key=lambda item: item["id"])
        if max_samples is not None:
            image_records = image_records[:max_samples]

        if not image_records:
            raise ValueError(
                "No usable images found. Check image paths, annotations, selected classes, "
                "or use --mock-data for a smoke test."
            )

        self.images = image_records
        self.annotations_by_image = annotations_by_image
        self.transforms = transforms or get_eval_transform()

    def resolve_image_path(self, file_name: str) -> Path:
        """Resolve a COCO file name relative to the configured image directory."""

        candidate = self.image_dir / file_name
        if candidate.is_file():
            return candidate
        return self.image_dir / Path(file_name).name

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Target]:
        image_info = self.images[index]
        image_path = self.resolve_image_path(image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        annotations = self.annotations_by_image.get(int(image_info["id"]), [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for annotation in annotations:
            x_min, y_min, width, height = annotation["bbox"]
            x_max = min(max(x_min + width, 0.0), float(image.width))
            y_max = min(max(y_min + height, 0.0), float(image.height))
            x_min = min(max(x_min, 0.0), float(image.width))
            y_min = min(max(y_min, 0.0), float(image.height))
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.category_id_to_label[annotation["category_id"]])
            areas.append(float((x_max - x_min) * (y_max - y_min)))
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        target: Target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(image_info["id"])], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        image_tensor, target = self.transforms(image, target)
        return image_tensor, target


class MockDetectionDataset(Dataset):
    """Tiny synthetic dataset for smoke-testing code paths without real images."""

    def __init__(
        self,
        num_images: int = 4,
        image_size: tuple[int, int] = (256, 256),
        num_classes: int = 3,
        transforms: Callable[[torch.Tensor, Target], tuple[torch.Tensor, Target]] | None = None,
    ) -> None:
        self.num_images = num_images
        self.height, self.width = image_size
        self.class_names = DEFAULT_TABLETOP_CLASSES[: max(1, min(num_classes, len(DEFAULT_TABLETOP_CLASSES)))]
        self.transforms = transforms

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Target]:
        generator = torch.Generator().manual_seed(index)
        image = torch.rand((3, self.height, self.width), generator=generator)
        offset = float((index * 13) % 40)
        box = torch.tensor(
            [[40.0 + offset, 45.0, min(170.0 + offset, self.width - 1.0), min(190.0, self.height - 1.0)]],
            dtype=torch.float32,
        )
        label = torch.tensor([(index % len(self.class_names)) + 1], dtype=torch.int64)
        target: Target = {
            "boxes": box,
            "labels": label,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]),
            "iscrowd": torch.zeros((1,), dtype=torch.int64),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

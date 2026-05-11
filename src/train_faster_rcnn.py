"""Training loop for TorchVision Faster R-CNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import CocoDetectionDataset, MockDetectionDataset, collate_fn, load_coco_class_names
from .models import build_faster_rcnn
from .transforms import get_train_transform
from .utils import ensure_dir, get_device, save_csv, save_json, set_seed


@dataclass
class FasterRCNNTrainingConfig:
    """Arguments needed to train Faster R-CNN."""

    data_dir: Path | None
    annotations: Path | None
    num_classes: int | None
    epochs: int
    batch_size: int
    lr: float
    device: str
    output_dir: Path
    num_workers: int
    mock_data: bool = False
    max_samples: int | None = None
    seed: int = 42


def train_one_epoch(model, dataloader, optimizer, device, epoch: int) -> float:
    """Train Faster R-CNN for one epoch and return mean loss."""

    model.train()
    running_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = float(losses.item())
        running_loss += loss_value
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return running_loss / max(len(dataloader), 1)


def _build_dataset(config: FasterRCNNTrainingConfig):
    if config.mock_data:
        foreground_classes = max((config.num_classes or 4) - 1, 1)
        dataset = MockDetectionDataset(num_images=max(4, config.batch_size * 2), num_classes=foreground_classes)
        class_names = dataset.class_names
        return dataset, class_names

    if config.data_dir is None or config.annotations is None:
        raise ValueError("`--data-dir` and `--annotations` are required unless `--mock-data` is used.")

    class_names = load_coco_class_names(config.annotations)
    dataset = CocoDetectionDataset(
        image_dir=config.data_dir,
        annotation_path=config.annotations,
        classes=class_names,
        transforms=get_train_transform(horizontal_flip=True),
        max_samples=config.max_samples,
    )
    return dataset, class_names


def run_training(config: FasterRCNNTrainingConfig) -> dict[str, Any]:
    """Train Faster R-CNN and save checkpoints plus a compact training log."""

    set_seed(config.seed)
    device = get_device(config.device)
    dataset, class_names = _build_dataset(config)
    inferred_num_classes = len(class_names) + 1
    num_classes = config.num_classes or inferred_num_classes
    if num_classes != inferred_num_classes:
        print(
            f"Warning: --num-classes={num_classes} differs from dataset-derived value "
            f"{inferred_num_classes}. Make sure this is intentional."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    model = build_faster_rcnn(num_classes=num_classes, pretrained=not config.mock_data)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=config.lr, momentum=0.9, weight_decay=0.0005)

    checkpoint_dir = ensure_dir(config.output_dir / "checkpoints")
    logs: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        mean_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        row = {"epoch": epoch, "train_loss": round(mean_loss, 6)}
        logs.append(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "train_loss": mean_loss,
        }
        torch.save(checkpoint, checkpoint_dir / f"faster_rcnn_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_dir / "faster_rcnn_latest.pth")

    save_csv(logs, config.output_dir / "training_log.csv")
    summary = {
        "model_name": "faster_rcnn_resnet50_fpn",
        "num_classes": num_classes,
        "class_names": class_names,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "device": str(device),
        "final_train_loss": logs[-1]["train_loss"] if logs else None,
        "checkpoint": str(checkpoint_dir / "faster_rcnn_latest.pth"),
    }
    save_json(summary, config.output_dir / "training_summary.json")
    return summary

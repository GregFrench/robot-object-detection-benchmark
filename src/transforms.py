"""Detection transforms for PIL images and TorchVision targets."""

from __future__ import annotations

import random
from typing import Callable

import torch
from PIL import Image
from torchvision.transforms import functional as F


Target = dict[str, torch.Tensor]
DetectionTransform = Callable[[Image.Image | torch.Tensor, Target], tuple[torch.Tensor, Target]]


class ComposeDetections:
    """Compose transforms that operate on both image and detection target."""

    def __init__(self, transforms: list[DetectionTransform]) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image | torch.Tensor, target: Target) -> tuple[torch.Tensor, Target]:
        for transform in self.transforms:
            image, target = transform(image, target)
        if not torch.is_tensor(image):
            image = F.to_tensor(image)
        return image, target


class ToTensorDetection:
    """Convert a PIL image to a float tensor while preserving target fields."""

    def __call__(self, image: Image.Image | torch.Tensor, target: Target) -> tuple[torch.Tensor, Target]:
        if torch.is_tensor(image):
            return image.float(), target
        return F.to_tensor(image), target


class RandomHorizontalFlipDetection:
    """Randomly flip an image and adjust xyxy boxes."""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image | torch.Tensor, target: Target) -> tuple[Image.Image | torch.Tensor, Target]:
        if random.random() >= self.probability:
            return image, target

        width = image.shape[-1] if torch.is_tensor(image) else image.width
        flipped = F.hflip(image)
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target = {**target, "boxes": boxes}
        return flipped, target


def get_train_transform(horizontal_flip: bool = True) -> ComposeDetections:
    """Build default training transforms."""

    transforms: list[DetectionTransform] = [ToTensorDetection()]
    if horizontal_flip:
        transforms.append(RandomHorizontalFlipDetection(0.5))
    return ComposeDetections(transforms)


def get_eval_transform() -> ComposeDetections:
    """Build deterministic evaluation transforms."""

    return ComposeDetections([ToTensorDetection()])

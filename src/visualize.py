"""Bounding-box visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .utils import ensure_dir


def image_to_pil(image: Image.Image | torch.Tensor | np.ndarray) -> Image.Image:
    """Convert common image representations to a PIL RGB image."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        tensor = image.detach().cpu().clamp(0, 1)
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if array.shape[2] == 1:
                array = array[:, :, 0]
            return Image.fromarray(array).convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def draw_predictions(
    image: Image.Image | torch.Tensor | np.ndarray,
    prediction: dict[str, Any],
    class_names: list[str],
    output_path: str | Path,
    confidence_threshold: float = 0.5,
    title: str | None = None,
) -> Path:
    """Draw prediction boxes and save a visualization image."""

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    pil_image = image_to_pil(image)

    boxes = prediction.get("boxes", [])
    labels = prediction.get("labels", [])
    scores = prediction.get("scores", [])
    boxes = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(pil_image)
    ax.axis("off")
    if title:
        ax.set_title(title)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < confidence_threshold:
            continue
        x_min, y_min, x_max, y_max = [float(value) for value in box]
        color = colors[int(label) % len(colors)]
        rectangle = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rectangle)
        label_name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        ax.text(
            x_min,
            max(0, y_min - 4),
            f"{label_name} {float(score):.2f}",
            color="white",
            fontsize=9,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
        )

    fig.tight_layout(pad=0)
    fig.savefig(output_file, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_file

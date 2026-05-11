"""Model builders for supported detection architectures."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_faster_rcnn(num_classes: int, pretrained: bool = True, trainable_backbone_layers: int | None = None):
    """Build Faster R-CNN with a task-specific classification head."""

    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if num_classes < 2:
        raise ValueError("num_classes must include background plus at least one object class.")

    try:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model_kwargs = {
            "weights": weights,
            "trainable_backbone_layers": trainable_backbone_layers,
        }
        if not pretrained:
            model_kwargs["weights_backbone"] = None
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**model_kwargs)
    except (AttributeError, TypeError):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_faster_rcnn_checkpoint(
    checkpoint_path: str | Path,
    num_classes: int | None,
    device,
    pretrained: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load a Faster R-CNN checkpoint saved by this project."""

    import torch

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    checkpoint_num_classes = checkpoint.get("num_classes") if isinstance(checkpoint, dict) else None
    inferred_num_classes = state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]
    resolved_num_classes = num_classes or checkpoint_num_classes or inferred_num_classes
    if resolved_num_classes != inferred_num_classes:
        raise ValueError(
            f"Checkpoint head has {inferred_num_classes} classes, but {resolved_num_classes} were requested."
        )
    model = build_faster_rcnn(num_classes=resolved_num_classes, pretrained=pretrained)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state_dict}

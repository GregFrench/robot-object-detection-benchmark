"""Simplified object detection evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import collate_fn
from .utils import ensure_dir, save_csv, save_json


@dataclass(frozen=True)
class EvaluationResult:
    """Summary for simplified IoU-based detection evaluation."""

    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two xyxy box arrays."""

    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, a_min=0, a_max=None)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.clip(union, a_min=1e-8, a_max=None)


def _target_to_record(target: dict[str, torch.Tensor]) -> dict[str, Any]:
    image_id = int(target["image_id"].view(-1)[0].item())
    return {
        "image_id": image_id,
        "boxes": target["boxes"].detach().cpu().numpy(),
        "labels": target["labels"].detach().cpu().numpy(),
    }


def _prediction_to_record(prediction: dict[str, torch.Tensor], image_id: int, score_threshold: float) -> dict[str, Any]:
    scores = prediction["scores"].detach().cpu().numpy()
    keep = scores >= score_threshold
    return {
        "image_id": image_id,
        "boxes": prediction["boxes"].detach().cpu().numpy()[keep],
        "labels": prediction["labels"].detach().cpu().numpy()[keep],
        "scores": scores[keep],
    }


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute area under the interpolated precision-recall curve."""

    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    class_names: list[str],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate detections with a simplified IoU-matching AP calculation.

    This is intentionally labelled as simplified mAP because it reports AP at
    one IoU threshold instead of the full COCO AP@[.50:.95] metric.
    """

    foreground_class_ids = range(1, len(class_names))
    targets_by_image = {int(target["image_id"]): target for target in targets}
    predictions_by_image = {int(prediction["image_id"]): prediction for prediction in predictions}

    class_metrics: list[dict[str, Any]] = []
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for class_id in foreground_class_ids:
        detections: list[tuple[int, float, np.ndarray]] = []
        gt_by_image: dict[int, np.ndarray] = {}

        for image_id, target in targets_by_image.items():
            target_labels = target["labels"]
            class_boxes = target["boxes"][target_labels == class_id]
            if len(class_boxes):
                gt_by_image[image_id] = class_boxes

        for image_id, prediction in predictions_by_image.items():
            scores = prediction["scores"]
            keep = (prediction["labels"] == class_id) & (scores >= score_threshold)
            for box, score in zip(prediction["boxes"][keep], scores[keep]):
                detections.append((image_id, float(score), box))

        detections.sort(key=lambda item: item[1], reverse=True)
        matched: dict[int, set[int]] = {image_id: set() for image_id in gt_by_image}
        tp = np.zeros(len(detections), dtype=np.float32)
        fp = np.zeros(len(detections), dtype=np.float32)
        n_gt = sum(len(boxes) for boxes in gt_by_image.values())

        for index, (image_id, _score, box) in enumerate(detections):
            gt_boxes = gt_by_image.get(image_id, np.zeros((0, 4), dtype=np.float32))
            if len(gt_boxes) == 0:
                fp[index] = 1
                continue

            ious = box_iou(np.asarray([box], dtype=np.float32), gt_boxes)[0]
            best_gt_index = int(np.argmax(ious)) if ious.size else -1
            best_iou = float(ious[best_gt_index]) if ious.size else 0.0
            if best_iou >= iou_threshold and best_gt_index not in matched[image_id]:
                tp[index] = 1
                matched[image_id].add(best_gt_index)
            else:
                fp[index] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(n_gt, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
        ap = _average_precision(recalls, precisions) if n_gt > 0 else 0.0

        true_positives = int(cum_tp[-1]) if len(cum_tp) else 0
        false_positives = int(cum_fp[-1]) if len(cum_fp) else 0
        total_tp += true_positives
        total_fp += false_positives
        total_gt += int(n_gt)
        class_metrics.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "ap": round(ap, 6),
                "precision": round(true_positives / max(true_positives + false_positives, 1), 6),
                "recall": round(true_positives / max(n_gt, 1), 6),
                "num_ground_truth": int(n_gt),
                "num_predictions": len(detections),
            }
        )

    valid_aps = [row["ap"] for row in class_metrics if row["num_ground_truth"] > 0]
    simplified_map = float(np.mean(valid_aps)) if valid_aps else 0.0
    summary = {
        "metric_type": "simplified_iou_ap",
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "simplified_map": round(simplified_map, 6),
        "precision": round(total_tp / max(total_tp + total_fp, 1), 6),
        "recall": round(total_tp / max(total_gt, 1), 6),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "ground_truth_objects": total_gt,
        "class_metrics": class_metrics,
    }
    return summary


def run_faster_rcnn_evaluation(
    model,
    dataset,
    device,
    batch_size: int = 1,
    num_workers: int = 0,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    model_name: str = "faster_rcnn",
) -> EvaluationResult:
    """Run inference and compute simplified detection metrics."""

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    model.eval()
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, batch_targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images = [image.to(device) for image in images]
            outputs = model(images)
            for output, target in zip(outputs, batch_targets):
                target_record = _target_to_record(target)
                prediction_record = _prediction_to_record(output, target_record["image_id"], score_threshold)
                predictions.append(prediction_record)
                targets.append(target_record)

    class_names = ["__background__", *dataset.class_names]
    metrics = evaluate_predictions(
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    metrics["model_name"] = model_name
    metrics["num_images"] = len(dataset)
    return EvaluationResult(metrics=metrics, predictions=predictions)


def save_evaluation_result(result: EvaluationResult, output_dir: str | Path, run_name: str) -> dict[str, Path]:
    """Save evaluation metrics as JSON and CSV."""

    output_path = ensure_dir(output_dir)
    json_path = save_json(result.metrics, output_path / f"{run_name}_metrics.json")
    summary_row = {
        key: value
        for key, value in result.metrics.items()
        if key not in {"class_metrics"}
    }
    csv_path = save_csv([summary_row], output_path / f"{run_name}_metrics.csv")
    class_csv_path = save_csv(result.metrics["class_metrics"], output_path / f"{run_name}_class_metrics.csv")
    return {"json": json_path, "csv": csv_path, "class_csv": class_csv_path}

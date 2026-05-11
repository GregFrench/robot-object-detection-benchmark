# Robotic Object Detection Benchmark

A PyTorch computer vision project benchmarking YOLO and Faster R-CNN for robotic object localization.

This repository is a clean, public educational/demo implementation of a tabletop object detection workflow. It is designed to look like the perception side of a robot pick-and-place or assistive robotics pipeline: localize common objects, evaluate detection quality, benchmark inference latency, and visualize predictions for inspection.

This project is a public companion project inspired by prior robotics computer vision work. It does not contain private research code or private datasets. The goal is to demonstrate the core ML engineering skills involved in robotic object detection: dataset preparation, model fine-tuning, evaluation, latency benchmarking, and prediction visualization.

## What This Demonstrates

- COCO-format dataset preparation for tabletop and household object classes
- Faster R-CNN fine-tuning with TorchVision detection models
- Optional YOLO training through the Ultralytics integration
- Simplified IoU-based object detection metrics: AP, mAP, precision, and recall
- Inference latency benchmarking for robotics-oriented model comparison
- Bounding-box visualization for qualitative inspection
- CLI-first experiment workflow with readable Python modules

## Why Detection Matters For Robotics

Object detection is often the first perception step before a robot can reason about the world. A 2D bounding box can feed downstream modules such as object tracking, depth lookup, 3D pose estimation, grasp candidate generation, human-assistive interaction, or task planning. For a robot, accuracy matters, but latency also matters because stale detections can lead to poor motion decisions.

## Pipeline

```text
COCO-format dataset
  -> tabletop class filtering
  -> Faster R-CNN / YOLO training
  -> detection evaluation
  -> latency benchmark
  -> bounding-box visualization
  -> robot perception use case
```

Default target classes:

```text
bottle, cup, bowl, fork, knife, spoon, apple, banana, book, cell phone
```

## Project Structure

```text
robot-object-detection-benchmark/
  configs/
  data/
  examples/
  src/
  scripts/
  outputs/
  README.md
  requirements.txt
```

## Installation

Create a Python 3.10+ environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

YOLO support is optional. If you want to train Ultralytics models:

```bash
python -m pip install ultralytics
```

GPU training is recommended for real experiments. CPU execution is supported for smoke tests and small debugging runs.

## Dataset Setup

This project does not download COCO automatically. Place COCO images and annotations wherever you keep local datasets, for example:

```text
/path/to/coco/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

The code also works with any custom COCO-format object detection dataset as long as the JSON contains `images`, `annotations`, and `categories`.

## Prepare A COCO Tabletop Subset

Prepare a training subset:

```bash
python scripts/prepare_coco_subset.py \
  --coco-images /path/to/coco/train2017 \
  --coco-annotations /path/to/coco/annotations/instances_train2017.json \
  --output-dir data/coco_tabletop_subset \
  --split-name train \
  --classes bottle cup bowl fork knife spoon apple banana book "cell phone" \
  --limit-images 1000 \
  --create-yolo-labels
```

Prepare a validation subset:

```bash
python scripts/prepare_coco_subset.py \
  --coco-images /path/to/coco/val2017 \
  --coco-annotations /path/to/coco/annotations/instances_val2017.json \
  --output-dir data/coco_tabletop_subset \
  --split-name val \
  --classes bottle cup bowl fork knife spoon apple banana book "cell phone" \
  --limit-images 250 \
  --create-yolo-labels
```

Preview without writing files:

```bash
python scripts/prepare_coco_subset.py \
  --coco-images /path/to/coco/train2017 \
  --coco-annotations /path/to/coco/annotations/instances_train2017.json \
  --output-dir data/coco_tabletop_subset \
  --dry-run
```

The script writes filtered COCO annotations, image copies or symlinks, optional YOLO labels, and a dataset summary.

## Train Faster R-CNN

```bash
python scripts/train_faster_rcnn.py \
  --data-dir data/coco_tabletop_subset/images/train \
  --annotations data/coco_tabletop_subset/annotations/instances_train.json \
  --epochs 5 \
  --batch-size 2 \
  --lr 0.005 \
  --device auto \
  --output-dir outputs \
  --num-workers 2
```

Checkpoints are saved to:

```text
outputs/checkpoints/
```

Smoke test without a real dataset:

```bash
python scripts/train_faster_rcnn.py --mock-data --epochs 1 --batch-size 1 --device cpu --num-workers 0
```

## Train YOLO

After preparing YOLO labels, use either the generated dataset YAML:

```bash
python scripts/train_yolo.py \
  --data-yaml data/coco_tabletop_subset/yolo_dataset.yaml \
  --model yolo11n.pt \
  --epochs 20 \
  --imgsz 640 \
  --batch-size 8
```

Or edit `configs/coco_tabletop_subset.yaml` and use it:

```bash
python scripts/train_yolo.py \
  --data-yaml configs/coco_tabletop_subset.yaml \
  --model yolo11n.pt \
  --epochs 20 \
  --imgsz 640
```

If `ultralytics` is not installed, the script exits with a clear installation message while the rest of the project remains usable.

## Evaluate Faster R-CNN

```bash
python scripts/evaluate_model.py \
  --checkpoint outputs/checkpoints/faster_rcnn_latest.pth \
  --data-dir data/coco_tabletop_subset/images/val \
  --annotations data/coco_tabletop_subset/annotations/instances_val.json \
  --device auto \
  --output-dir outputs/metrics \
  --score-threshold 0.05 \
  --iou-threshold 0.5
```

This implementation reports a simplified AP/mAP at one IoU threshold. It is useful for educational benchmarking and quick model comparison, but it is not a replacement for full COCO AP@[.50:.95].

Example metrics JSON:

```json
{
  "metric_type": "simplified_iou_ap",
  "iou_threshold": 0.5,
  "simplified_map": 0.421,
  "precision": 0.68,
  "recall": 0.55,
  "model_name": "faster_rcnn",
  "num_images": 250
}
```

## Compare Models

```bash
python scripts/compare_models.py \
  --metrics-dir outputs/metrics \
  --output outputs/model_comparison.csv
```

The comparison table includes model name, mAP or simplified mAP, precision, recall, latency when available, notes, and source file.

## Visualize Predictions

Run Faster R-CNN on validation samples and save annotated images:

```bash
python scripts/visualize_predictions.py \
  --checkpoint outputs/checkpoints/faster_rcnn_latest.pth \
  --data-dir data/coco_tabletop_subset/images/val \
  --annotations data/coco_tabletop_subset/annotations/instances_val.json \
  --confidence-threshold 0.5 \
  --max-images 8 \
  --output-dir outputs/sample_predictions
```

Visualizations are written to:

```text
outputs/sample_predictions/
```

## Benchmark Latency

```bash
python scripts/benchmark_latency.py \
  --checkpoint outputs/checkpoints/faster_rcnn_latest.pth \
  --data-dir data/coco_tabletop_subset/images/val \
  --annotations data/coco_tabletop_subset/annotations/instances_val.json \
  --num-images 50 \
  --warmup 5 \
  --device auto \
  --output-dir outputs/latency
```

Example latency JSON:

```json
{
  "mean_latency_ms": 42.7,
  "median_latency_ms": 41.9,
  "images_per_second": 23.4,
  "device": "cuda:0"
}
```

## Script Reference

Each script supports `--help`:

```bash
python scripts/prepare_coco_subset.py --help
python scripts/train_faster_rcnn.py --help
python scripts/train_yolo.py --help
python scripts/evaluate_model.py --help
python scripts/compare_models.py --help
python scripts/visualize_predictions.py --help
python scripts/benchmark_latency.py --help
```

## Limitations

- The included mAP implementation is simplified and uses one IoU threshold.
- The project benchmarks 2D object localization, not full 6D pose estimation.
- COCO tabletop classes are a proxy for robotics scenes and may not match a real robot camera distribution.
- YOLO metrics are not automatically normalized into the project metric schema; export or copy relevant results into `outputs/metrics/` for comparison.
- No large datasets, trained weights, or experiment artifacts are committed.

## Future Improvements

- Add SSD or RT-DETR as another detector family
- Add segmentation masks for object contours
- Add depth camera input for 3D localization
- Add ROS 2 integration for publishing detections
- Add real-time camera inference
- Add robot grasp-point estimation from detections
- Deploy models with ONNX or TensorRT

## Publication Notes

The repository is safe to publish publicly. Dataset artifacts, COCO images, checkpoints, runs, and generated outputs are ignored by default. Keep private datasets and institutional code outside this repo.

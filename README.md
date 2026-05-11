# Robotic Object Detection Benchmark

A PyTorch computer vision project benchmarking YOLO and Faster R-CNN for robotic object localization.

## What This Demonstrates

- COCO-format dataset preparation for tabletop and household object classes
- Faster R-CNN fine-tuning with TorchVision detection models
- Optional YOLO training through the Ultralytics integration
- VOC-style AP at a configurable IoU threshold, plus precision and recall
- Inference latency benchmarking for robotics-oriented model comparison
- Bounding-box visualization for qualitative inspection
- CLI-first experiment workflow with readable Python modules

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

The code also works with any custom COCO-format object detection dataset as long as the JSON contains `images`, `annotations`, and `categories`. Class order is preserved from the COCO `categories` list, and Faster R-CNN checkpoints store the class order used during training so validation labels can be remapped consistently.

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
  --shuffle \
  --seed 42 \
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

The script writes filtered COCO annotations, image copies or symlinks, optional YOLO labels, and a dataset summary. COCO boxes are clipped to image bounds before YOLO conversion so normalized labels stay within the expected image coordinate frame.

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

This implementation reports VOC-style AP at one IoU threshold. With `--iou-threshold 0.5`, the summary field `map50` is AP50 averaged across classes with ground-truth examples. It is useful for educational benchmarking and quick model comparison, but it is not COCO AP@[.50:.95] and does not include COCO area ranges, max detection caps, or crowd-region handling.

Example metrics JSON:

```json
{
  "metric_type": "voc_style_ap_at_single_iou",
  "iou_threshold": 0.5,
  "map50": 0.421,
  "map_at_iou": 0.421,
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

The comparison table includes model name, mAP or AP-at-IoU, precision, recall, latency when available, notes, and source file. If metric and latency JSON files share the same `model_name`, the comparison script merges them into one row.

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

You can also visualize the prediction JSON written by evaluation:

```bash
python scripts/visualize_predictions.py \
  --predictions outputs/metrics/faster_rcnn_predictions.json \
  --confidence-threshold 0.5 \
  --max-images 8
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

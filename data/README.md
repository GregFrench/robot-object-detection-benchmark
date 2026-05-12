# Data Directory

This repository does not download COCO or any large dataset automatically.

Place local data here only for experiments. Recommended layout after preparing a subset:

```text
data/
  coco_tabletop_subset/
    annotations/
      instances_train.json
      instances_val.json
    images/
      train/
      val/
    labels/
      train/
      val/
    dataset_summary_train.json
    dataset_summary_val.json
    yolo_dataset.yaml
```

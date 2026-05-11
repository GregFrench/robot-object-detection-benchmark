# Sample Predictions

Prediction visualizations are written to:

```text
outputs/sample_predictions/
```

Each image contains predicted bounding boxes, class labels, and confidence
scores. In a robotics perception pipeline, these boxes would typically be
passed downstream to object tracking, 3D pose estimation, grasp candidate
generation, or a pick-and-place planner.

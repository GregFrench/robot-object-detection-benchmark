"""Inference latency benchmarking helpers."""

from __future__ import annotations

import statistics
import time
from itertools import cycle, islice
from typing import Any

import torch


def _sync_if_needed(device) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    model,
    images: list[torch.Tensor],
    device,
    warmup: int = 5,
    iterations: int | None = None,
) -> dict[str, Any]:
    """Measure single-image inference latency for a TorchVision detector."""

    if not images:
        raise ValueError("At least one image is required for latency benchmarking.")

    iterations = iterations or len(images)
    model.eval()
    image_stream = cycle(images)

    with torch.no_grad():
        for image in islice(image_stream, warmup):
            model([image.to(device)])
        _sync_if_needed(device)

        timings_ms: list[float] = []
        for image in islice(image_stream, iterations):
            image = image.to(device)
            _sync_if_needed(device)
            start = time.perf_counter()
            model([image])
            _sync_if_needed(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    return {
        "mean_latency_ms": round(mean_ms, 4),
        "median_latency_ms": round(median_ms, 4),
        "images_per_second": round(1000.0 / mean_ms, 4) if mean_ms > 0 else 0.0,
        "num_iterations": iterations,
        "warmup_iterations": warmup,
        "device": str(device),
    }

"""Recent-detail time mapping and peak-preserving display summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass


def history_position(age: float, retention: float, detail: float) -> float:
    """Normalized left-to-right position: older history left, current reading right."""
    retention = max(1.0, retention)
    detail = min(retention, max(1.0, detail))
    age = min(retention, max(0.0, age))
    if detail >= retention:
        return 1.0 - age / retention
    recent = detail / 2
    if age <= recent:
        return 1.0 - age / detail
    compression = retention / detail - 1
    fraction = (age - recent) / (retention - recent)
    return 0.5 * (1.0 - math.log1p(compression * fraction) / math.log1p(compression))


def history_span(times, retention: float, detail: float) -> float:
    """Fill the initial linear window before gradually compressing older readings."""
    elapsed = max(0.0, times[-1] - times[0]) if times else 0.0
    return min(retention, max(detail, elapsed))


@dataclass(frozen=True)
class HistoryBin:
    x: float
    cpu_mean: float
    cpu_min: float
    cpu_max: float
    gpu_mean: float
    gpu_min: float
    gpu_max: float
    processing: bool
    count: int


def summarize_history(samples, times, retention: float, detail: float, width: int):
    """Group subpixel readings without discarding spikes or processing intervals.

    Means describe the sampled readings, rather than interpolating unobserved activity.
    Raw history remains separate so zooming can recompute summaries at higher resolution.
    """
    if not samples:
        return []
    if len(times) != len(samples):
        raise ValueError("Every metrics reading needs a sample time")
    width = max(1, width)
    now = times[-1]
    span = history_span(times, retention, detail)
    buckets = {}
    for (cpu, gpu, active), timestamp in zip(samples, times):
        age = max(0.0, now - timestamp)
        if age > retention:
            continue
        x = min(width, max(0, round(history_position(age, span, detail) * width)))
        if x not in buckets:
            buckets[x] = [cpu, cpu, cpu, gpu, gpu, gpu, bool(active), 1]
        else:
            values = buckets[x]
            values[0] += cpu
            values[1] = min(values[1], cpu)
            values[2] = max(values[2], cpu)
            values[3] += gpu
            values[4] = min(values[4], gpu)
            values[5] = max(values[5], gpu)
            values[6] = values[6] or active
            values[7] += 1
    return [
        HistoryBin(x, cpu_sum / count, cpu_min, cpu_max,
                   gpu_sum / count, gpu_min, gpu_max, bool(active), count)
        for x, (cpu_sum, cpu_min, cpu_max, gpu_sum, gpu_min, gpu_max, active, count)
        in sorted(buckets.items())
    ]

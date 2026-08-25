"""Metrics export (spec section 43): canonical metric names, Prometheus text
format, and a tiny in-process registry fed by the pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

CANONICAL_METRICS = [
    "camera_model",
    "calibration_source",
    "calibration_id",
    "projection_roundtrip_p95_px",
    "projection_roundtrip_p99_px",
    "valid_pixel_ratio",
    "stereo_epipolar_median_px",
    "stereo_epipolar_p95_px",
    "virtual_camera_coverage",
    "feature_retention",
    "calibration_health_score",
    "runtime_ms",
    "gpu_memory_mb",
]


def sanitize_name(name: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def to_prometheus(metrics: dict[str, Any], job: str = "camera_geometry") -> str:
    """Render a flat metric dict (values may be numeric or nested timing dicts)
    as Prometheus text exposition. Non-numeric values become labels on a
    single `camera_geometry_info` gauge."""
    lines = ["# TYPE camera_geometry_info gauge"]

    def walk(prefix: str, value: Any):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}_{sanitize_name(k)}" if prefix else sanitize_name(k), v)
        elif isinstance(value, bool):
            lines.append(f"{prefix} {int(value)}")
        elif isinstance(value, (int, float)):
            lines.append(f"{prefix} {value}")
        else:
            label = str(value).replace('"', "'").replace("\n", " ")
            lines.append(f'{prefix}{{value="{label}"}} 1')

    lines.append(f'camera_geometry_info{{job="{job}"}} 1')
    for k, v in metrics.items():
        walk(sanitize_name(k), v)
    return "\n".join(lines) + "\n"


@dataclass
class StageTimer:
    """Context manager recording per-stage wall time into a dict (ms)."""

    sink: dict
    stage: str
    _t0: float = field(default=0.0, repr=False)

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink.setdefault("runtime_ms", {})[f"{self.stage}_ms"] = int(
            (time.perf_counter() - self._t0) * 1000
        )
        return False

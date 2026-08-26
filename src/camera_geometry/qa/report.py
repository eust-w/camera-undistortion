"""Health score aggregation and QA report building.

S = 0.15*S_schema + 0.20*S_projection + 0.25*S_stereo + 0.25*S_multiview + 0.15*S_temporal

Categories not evaluated (e.g. multiview/temporal in MVP) are excluded and the
remaining weights renormalized; they are flagged in the report.
PASS: >= 0.90, DEGRADED: >= 0.75, FAIL: < 0.75
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .checks import CheckResult

WEIGHTS = {
    "schema": 0.15,
    "projection": 0.20,
    "stereo": 0.25,
    "multiview": 0.25,
    "temporal": 0.15,
}

CHECK_TO_CATEGORY = {
    "schema": "schema",
    "rotation": "schema",
    "baseline": "schema",
    "projection_roundtrip": "projection",
    "valid_coverage": "projection",
    "stereo_epipolar": "stereo",
}

PASS_THRESHOLD = 0.90
DEGRADED_THRESHOLD = 0.75


@dataclass
class HealthReport:
    score: float
    status: str
    categories: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "score": round(self.score, 4),
            "status": self.status,
            "categories": self.categories,
            "weights_used": {k: v for k, v in WEIGHTS.items() if k in self.categories},
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "score": round(c.score, 4),
                    "evaluated": c.evaluated,
                    "metrics": c.metrics,
                }
                for c in self.checks
            ],
        }


def aggregate_health(checks: list[CheckResult]) -> HealthReport:
    cat_scores: dict[str, list[float]] = {}
    for check in checks:
        cat = CHECK_TO_CATEGORY.get(check.name)
        if cat is None or not check.evaluated:
            continue
        cat_scores.setdefault(cat, []).append(check.score)

    categories = {}
    used_weight_sum = 0.0
    score = 0.0
    for cat, weight in WEIGHTS.items():
        if cat in cat_scores:
            s = float(np_min(cat_scores[cat]))
            categories[cat] = {"score": round(s, 4), "weight": weight}
            used_weight_sum += weight
            score += weight * s
        else:
            categories[cat] = {"score": None, "weight": weight, "note": "not_evaluated"}  # type: ignore[dict-item]

    final = score / used_weight_sum if used_weight_sum > 0 else 0.0
    if final >= PASS_THRESHOLD:
        status = "PASS"
    elif final >= DEGRADED_THRESHOLD:
        status = "DEGRADED"
    else:
        status = "FAIL"
    hard_fail = any(c.evaluated and not c.passed and c.name in ("schema",) for c in checks)
    if hard_fail:
        status = "FAIL"
        final = min(final, 0.5)
    return HealthReport(score=final, status=status, categories=categories, checks=checks)


def np_min(values):
    import numpy as np

    return float(np.min(values))


def build_metrics_section(extra: dict | None = None) -> dict[str, Any]:
    """Canonical metrics block (section 43 of the spec)."""
    metrics: dict[str, Any] = {
        "camera_model": None,
        "calibration_source": None,
        "calibration_id": None,
        "projection_roundtrip_p95": None,
        "projection_roundtrip_p99": None,
        "valid_pixel_ratio": None,
        "stereo_epipolar_median": None,
        "stereo_epipolar_p95": None,
        "virtual_camera_coverage": None,
        "feature_retention": None,
        "calibration_health_score": None,
        "runtime_ms": {},
        "gpu_memory_mb": None,
    }
    if extra:
        metrics.update({k: v for k, v in extra.items()})
    return metrics

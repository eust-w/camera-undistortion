"""Calibration validation orchestrator: static QA without images.

Deliberately tolerant of malformed input: QA measures and reports problems
(e.g. non-orthonormal rotation) instead of crashing, so the state machine can
route to FALLBACK / QUARANTINE.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.factory import build_camera
from ..qa.checks import (
    CheckResult,
    check_baseline,
    check_projection_roundtrip,
    check_rotation,
    check_schema,
)
from ..qa.report import HealthReport, aggregate_health
from ..schema import CameraEntry, DeviceCalibration


@dataclass
class CalibrationQA:
    calibration_id: str
    report: HealthReport

    @property
    def status(self) -> str:
        return self.report.status

    @property
    def score(self) -> float:
        return self.report.score


def validate_camera(
    calib: DeviceCalibration,
    entry: CameraEntry,
    baseline_range: tuple[float, float] = (0.10, 0.14),
    roundtrip_samples: int = 100_000,
) -> list[CheckResult]:
    checks: list[CheckResult] = [check_schema(calib, entry)]

    ext = entry.extrinsics
    if ext is not None:
        checks.append(check_rotation(ext.rotation))
        if entry.id != ext.reference_camera:
            checks.append(check_baseline(ext.rotation, ext.translation_m, baseline_range))
        else:
            checks.append(CheckResult.not_evaluated("baseline", "reference camera"))
    else:
        checks.append(CheckResult.not_evaluated("rotation", "no extrinsics for this camera"))
        checks.append(CheckResult.not_evaluated("baseline", "no extrinsics for this camera"))

    try:
        camera = build_camera(entry)
        checks.append(check_projection_roundtrip(camera, n_samples=roundtrip_samples))
    except Exception as exc:
        checks.append(
            CheckResult("projection_roundtrip", False, 0.0, {"error": f"camera build failed: {exc}"})
        )
    return checks


def validate_calibration(
    calib: DeviceCalibration,
    baseline_range: tuple[float, float] = (0.10, 0.14),
    roundtrip_samples: int = 100_000,
) -> dict[str, CalibrationQA]:
    """Validate every camera in the device calibration. Returns per-camera QA."""
    results: dict[str, CalibrationQA] = {}
    for entry in calib.cameras:
        checks = validate_camera(
            calib, entry, baseline_range=baseline_range, roundtrip_samples=roundtrip_samples
        )
        results[entry.id] = CalibrationQA(
            calibration_id=calib.calibration_id(entry.id), report=aggregate_health(checks)
        )
    return results

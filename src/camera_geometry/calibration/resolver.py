"""Calibration resolution with explicit priority order.

1. current device Factory Calibration (config file)
2. validated Device Calibration (registry, QA-passed refinement)
3. same-model Nominal Calibration
4. multi-view refinement / AnyCalib  -> Phase 5 stubs, raise FallbackRequired
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import DeviceCalibration
from .store import CalibrationStore


class CalibrationNotFound(RuntimeError):
    pass


class FallbackRequired(RuntimeError):
    """Raised when resolution cannot complete and the state machine must fall back."""

    def __init__(self, stage: str, reason: str):
        super().__init__(f"[{stage}] {reason}")
        self.stage = stage
        self.reason = reason


@dataclass
class ResolvedCalibration:
    calibration: DeviceCalibration
    camera_id: str
    calibration_id: str
    source: str
    quality_status: str = "unvalidated"
    priority: int = 0
    notes: list = field(default_factory=list)

    @property
    def entry(self):
        return self.calibration.camera(self.camera_id)


SOURCE_PRIORITY = {
    "factory": 0,
    "validated_device": 1,
    "nominal": 2,
    "multiview_refinement": 3,
    "anycalib": 4,
}


class CalibrationResolver:
    def __init__(self, device_calibration: DeviceCalibration | None = None, store: CalibrationStore | None = None):
        self.device_calibration = device_calibration
        self.store = store

    def resolve(self, camera_id: str, resolution: tuple[int, int] | None = None) -> ResolvedCalibration:
        candidates: list[ResolvedCalibration] = []

        if self.device_calibration is not None:
            try:
                entry = self.device_calibration.camera(camera_id)
                if resolution is None or entry.resolution.wh == tuple(resolution):
                    src_label = self.device_calibration.calibration.source
                    candidates.append(
                        ResolvedCalibration(
                            calibration=self.device_calibration,
                            camera_id=camera_id,
                            calibration_id=self.device_calibration.calibration_id(camera_id),
                            source=src_label,
                            quality_status=self.device_calibration.quality.status,
                            priority=SOURCE_PRIORITY.get(src_label, SOURCE_PRIORITY["factory"]),
                        )
                    )
            except KeyError:
                pass

        if self.store is not None:
            for record in self.store.list_records():
                calib = record.calibration
                if record.quality_status != "PASS":
                    continue
                try:
                    entry = calib.camera(camera_id)
                except KeyError:
                    continue
                if resolution is not None and entry.resolution.wh != tuple(resolution):
                    continue
                embedded = calib.calibration.source or ""
                source = embedded if embedded not in ("factory", "unvalidated", "") else "validated_device"
                candidates.append(
                    ResolvedCalibration(
                        calibration=calib,
                        camera_id=camera_id,
                        calibration_id=record.calibration_id,
                        source=source,
                        quality_status=record.quality_status,
                        priority=SOURCE_PRIORITY["validated_device"],
                        notes=[f"registry:{record.calibration_id}"],
                    )
                )

        if not candidates:
            raise CalibrationNotFound(
                f"no calibration for camera '{camera_id}'"
                + (f" at resolution {resolution}" if resolution else "")
            )

        best = max(candidates, key=lambda c: c.priority)
        if best.source == "factory" and self.store is not None:
            for cand in candidates:
                if cand.source == "validated_device":
                    best.notes.append(f"superseded registry candidate: {cand.calibration_id}")
        return best

    def request_fallback(self, stage: str) -> None:
        raise FallbackRequired(stage, "no usable calibration; refinement/anycalib not available in MVP")

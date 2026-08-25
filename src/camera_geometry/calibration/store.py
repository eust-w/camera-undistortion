"""Persistent calibration registry.

Calibration ID: {serial}:{camera}:{WxH}:{version}
Episodes only store references; full calibration lives here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import DeviceCalibration


def _safe_id(calibration_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", calibration_id)


@dataclass
class CalibrationRecord:
    calibration_id: str
    calibration: DeviceCalibration
    quality_status: str = "unvalidated"
    health_score: float | None = None
    qa_report: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "calibration_id": self.calibration_id,
            "calibration": self.calibration.model_dump(),
            "quality_status": self.quality_status,
            "health_score": self.health_score,
            "qa_report": self.qa_report,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "CalibrationRecord":
        return cls(
            calibration_id=payload["calibration_id"],
            calibration=DeviceCalibration.model_validate(payload["calibration"]),
            quality_status=payload.get("quality_status", "unvalidated"),
            health_score=payload.get("health_score"),
            qa_report=payload.get("qa_report", {}),
        )


class CalibrationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "calibrations").mkdir(parents=True, exist_ok=True)

    def _path(self, calibration_id: str) -> Path:
        return self.root / "calibrations" / f"{_safe_id(calibration_id)}.json"

    def add(self, record: CalibrationRecord) -> CalibrationRecord:
        self._path(record.calibration_id).write_text(
            json.dumps(record.to_json(), indent=2), encoding="utf-8"
        )
        return record

    def get(self, calibration_id: str) -> CalibrationRecord:
        path = self._path(calibration_id)
        if not path.exists():
            raise KeyError(f"unknown calibration_id '{calibration_id}'")
        return CalibrationRecord.from_json(json.loads(path.read_text(encoding="utf-8")))

    def exists(self, calibration_id: str) -> bool:
        return self._path(calibration_id).exists()

    def update_qa(self, calibration_id: str, status: str, score: float | None, report: dict) -> None:
        rec = self.get(calibration_id)
        rec.quality_status = status
        rec.health_score = score
        rec.qa_report = report
        rec.calibration.quality.status = status
        rec.calibration.quality.health_score = score
        self.add(rec)

    def list_records(self) -> list[CalibrationRecord]:
        out = []
        for path in sorted((self.root / "calibrations").glob("*.json")):
            try:
                out.append(CalibrationRecord.from_json(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return out

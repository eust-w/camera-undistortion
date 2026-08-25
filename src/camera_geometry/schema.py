"""Canonical camera calibration schema (camera-geometry/v1).

All business code must consume this schema, never raw vendor calibration.
Units: distance=meter, time=nanosecond, rotation=radian / rotation matrix,
image=pixel.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "camera-geometry/v1"
CODE_VERSION = "0.1.0"


class Resolution(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def wh(self) -> tuple[int, int]:
        return self.width, self.height


class Intrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class KB4Distortion(BaseModel):
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0


class ModelSpec(BaseModel):
    type: Literal["kannala_brandt_4", "pinhole", "eucm", "double_sphere"]
    convention: str = "opencv_fisheye_kb4"
    params: dict[str, float] = Field(default_factory=dict)


class Extrinsics(BaseModel):
    """T_ref_self: transforms points from this camera frame to reference frame.

    P_ref = R @ P_self + t.
    """

    reference_camera: str
    rotation: List[List[float]]
    translation_m: List[float]

    @field_validator("rotation")
    @classmethod
    def _check_rotation(cls, v: List[List[float]]) -> List[List[float]]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        return v

    @field_validator("translation_m")
    @classmethod
    def _check_translation(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("translation_m must have 3 components")
        return v


class ShutterInfo(BaseModel):
    type: str = "unknown"
    readout_time_ns: Optional[int] = None


class DeviceInfo(BaseModel):
    vendor: str = "unknown"
    serial_number: str = "unknown"


class CalibrationMeta(BaseModel):
    source: str = "factory"
    version: str = "v1"


class QualityInfo(BaseModel):
    status: str = "unvalidated"
    health_score: Optional[float] = None


class CameraEntry(BaseModel):
    id: str
    resolution: Resolution
    model: ModelSpec
    intrinsics: Intrinsics
    distortion: Optional[KB4Distortion] = None
    extrinsics: Optional[Extrinsics] = None


class DeviceCalibration(BaseModel):
    schema_version: str = SCHEMA_VERSION
    device: DeviceInfo = DeviceInfo()
    cameras: List[CameraEntry]
    shutter: ShutterInfo = ShutterInfo()
    calibration: CalibrationMeta = CalibrationMeta()
    quality: QualityInfo = QualityInfo()

    def camera(self, camera_id: str) -> CameraEntry:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise KeyError(f"camera '{camera_id}' not in device calibration")

    def calibration_id(self, camera_id: str) -> str:
        cam = self.camera(camera_id)
        serial = self.device.serial_number or "unknown"
        w, h = cam.resolution.wh
        return f"{serial}:{camera_id}:{w}x{h}:{self.calibration.version}"


def make_calibration_id(serial: str, camera_id: str, width: int, height: int, version: str = "v1") -> str:
    return f"{serial}:{camera_id}:{width}x{height}:{version}"

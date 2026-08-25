"""Build CameraModel instances from the canonical schema."""

from __future__ import annotations

from ..schema import CameraEntry
from .base import CameraModel
from .double_sphere import DoubleSphere
from .eucm import EUCM
from .kb4 import KannalaBrandt4
from .pinhole import PinholeCamera


def build_camera(entry: CameraEntry) -> CameraModel:
    intr = entry.intrinsics
    params = entry.model.params or {}
    w, h = entry.resolution.wh
    if entry.model.type == "pinhole":
        return PinholeCamera(intr.fx, intr.fy, intr.cx, intr.cy, w, h)
    if entry.model.type == "kannala_brandt_4":
        d = entry.distortion or (0.0, 0.0, 0.0, 0.0)
        return KannalaBrandt4(
            intr.fx, intr.fy, intr.cx, intr.cy,
            getattr(d, "k1", 0.0), getattr(d, "k2", 0.0),
            getattr(d, "k3", 0.0), getattr(d, "k4", 0.0), w, h,
        )
    if entry.model.type == "eucm":
        return EUCM(intr.fx, intr.fy, intr.cx, intr.cy,
                    float(params.get("alpha", 0.5)), w, h)
    if entry.model.type == "double_sphere":
        return DoubleSphere(intr.fx, intr.fy, intr.cx, intr.cy,
                            float(params.get("xi", 0.5)), float(params.get("alpha", 0.5)), w, h)
    raise ValueError(f"unsupported camera model type: {entry.model.type}")

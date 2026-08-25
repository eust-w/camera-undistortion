"""Ideal pinhole camera (used for virtual / rectified cameras)."""

from __future__ import annotations

import numpy as np

from .base import CameraModel


class PinholeCamera(CameraModel):
    model_type = "pinhole"

    def __init__(self, fx, fy, cx, cy, width: int, height: int):
        super().__init__(width, height)
        if min(fx, fy) <= 0:
            raise ValueError("fx/fy must be positive")
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64
        )

    @staticmethod
    def from_fov(width: int, height: int, fov_x_deg: float) -> "PinholeCamera":
        fx = (width / 2.0) / np.tan(np.radians(fov_x_deg) / 2.0)
        return PinholeCamera(fx, fx, (width - 1) / 2.0, (height - 1) / 2.0, width, height)

    def _project(self, xyz: np.ndarray) -> np.ndarray:
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * X / Z + self.cx
            v = self.fy * Y / Z + self.cy
        return np.stack([u, v], axis=-1)

    def _unproject(self, uv: np.ndarray) -> np.ndarray:
        xd = (uv[..., 0] - self.cx) / self.fx
        yd = (uv[..., 1] - self.cy) / self.fy
        rays = np.stack([xd, yd, np.ones_like(xd)], axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def valid_mask(self) -> np.ndarray:
        return np.ones((self.height, self.width), dtype=bool)

    def intrinsics_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

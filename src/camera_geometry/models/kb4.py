"""Kannala-Brandt 4 (equidistant fisheye, OpenCV fisheye convention).

theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
u = fx * (X/r) * theta_d + cx , r = sqrt(X^2+Y^2), theta = atan2(r, Z)

Unprojection solves theta from theta_d by vectorized Newton iteration,
then the ray is [xd*s, yd*s, 1] with s = tan(theta)/theta_d, normalized.
"""

from __future__ import annotations

import numpy as np

from .base import CameraModel, to_numpy

_EPS_R = 1e-12
_MAX_THETA = np.pi / 2 - 1e-6
_NEWTON_TOL = 1e-12
_NEWTON_ITERS = 50


def solve_theta(theta_d: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Vectorized Newton solve of theta_d = theta + k1 t^3 + k2 t^5 + k3 t^7 + k4 t^9."""
    theta = theta_d.copy()
    for _ in range(_NEWTON_ITERS):
        th2 = theta * theta
        f = theta * (1.0 + th2 * (k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3])))) - theta_d
        fp = 1.0 + th2 * (3 * k[0] + th2 * (5 * k[1] + th2 * (7 * k[2] + th2 * 9 * k[3])))
        delta = f / np.maximum(fp, 1e-6)
        delta = np.clip(delta, -0.5, 0.5)
        theta = np.clip(theta - delta, 0.0, np.pi)
        if float(np.max(np.abs(delta), initial=0.0)) < _NEWTON_TOL:
            break
    return theta


class KannalaBrandt4(CameraModel):
    model_type = "kannala_brandt_4"

    def __init__(self, fx, fy, cx, cy, k1, k2, k3, k4, width: int, height: int):
        super().__init__(width, height)
        if min(fx, fy) <= 0:
            raise ValueError("fx/fy must be positive")
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.k = np.array([k1, k2, k3, k4], dtype=np.float64)

    # ---- projection -------------------------------------------------
    def _project(self, xyz: np.ndarray) -> np.ndarray:
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        r = np.hypot(X, Y)
        theta = np.arctan2(r, Z)
        th2 = theta * theta
        theta_d = theta * (
            1.0 + th2 * (self.k[0] + th2 * (self.k[1] + th2 * (self.k[2] + th2 * self.k[3])))
        )
        scale = theta_d / np.maximum(r, _EPS_R)
        u = self.fx * X * scale + self.cx
        v = self.fy * Y * scale + self.cy
        return np.stack([u, v], axis=-1)

    def _unproject(self, uv: np.ndarray) -> np.ndarray:
        xd = (uv[..., 0] - self.cx) / self.fx
        yd = (uv[..., 1] - self.cy) / self.fy
        td = np.hypot(xd, yd)
        theta = solve_theta(td, self.k)
        safe_td = np.where(td > _EPS_R, td, 1.0)
        s = np.tan(theta) / safe_td
        rays = np.stack([xd * s, yd * s, np.ones_like(td)], axis=-1)
        center = td <= _EPS_R
        if np.any(center):
            rays[center] = [0.0, 0.0, 1.0]
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays / np.maximum(norms, _EPS_R)

    # ---- validity ----------------------------------------------------
    def max_theta(self) -> float:
        probes = [
            (0, self.height / 2),
            (self.width - 1, self.height / 2),
            (self.width / 2, 0),
            (self.width / 2, self.height - 1),
            (0, 0),
            (self.width - 1, self.height - 1),
        ]
        thetas = []
        for u, v in probes:
            d = self._unproject(np.array([[u, v]], dtype=np.float64))[0]
            if np.all(np.isfinite(d)):
                thetas.append(float(np.arccos(np.clip(d[2], -1, 1))))
        return max(thetas) if thetas else _MAX_THETA

    def theta_of_pixels(self, uv: np.ndarray) -> np.ndarray:
        rays = self.unproject(uv)
        return np.arccos(np.clip(to_numpy(rays)[..., 2], -1.0, 1.0))

    def valid_mask(self, roundtrip_tol_px: float = 0.25) -> np.ndarray:
        uu, vv = self.pixel_grid()
        uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        rays = self._unproject(uv)
        finite = np.all(np.isfinite(rays), axis=-1)
        front = rays[:, 2] > np.cos(_MAX_THETA)
        reproj = np.full_like(uv, np.nan)
        if np.any(finite):
            reproj[finite] = self._project(rays[finite])
        err = np.linalg.norm(reproj - uv, axis=-1)
        ok = finite & front & (err < roundtrip_tol_px)
        return ok.reshape(self.height, self.width)

    def fov_deg(self) -> float:
        return float(np.degrees(2.0 * self.max_theta()))

    # ---- introspection -------------------------------------------------
    def intrinsics_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def distortion_dict(self) -> dict | None:
        return {f"k{i+1}": float(v) for i, v in enumerate(self.k)}

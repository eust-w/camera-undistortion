"""Extended Unified Camera Model (Khomutenko et al., 2016).

Params: fx, fy, cx, cy, alpha (0 <= alpha < 1).
  d = sqrt(alpha*r^2 + (1-alpha)*z^2), r^2 = x^2+y^2
  u = fx*x/d + cx ,  v = fy*y/d + cy
alpha=0 reduces to pinhole; alpha->1 degenerates (z unobservable), so rejected.
Valid domain: rho = ||(u-cx)/fx,(v-cy)/fy|| < 1/sqrt(alpha).
"""

from __future__ import annotations

import numpy as np

from .base import CameraModel


class EUCM(CameraModel):
    model_type = "eucm"

    def __init__(self, fx, fy, cx, cy, alpha: float, width: int, height: int):
        super().__init__(width, height)
        if min(fx, fy) <= 0:
            raise ValueError("fx/fy must be positive")
        if not (0.0 <= alpha < 1.0):
            raise ValueError("EUCM alpha must be in [0, 1)")
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.alpha = float(alpha)

    def _project(self, xyz: np.ndarray) -> np.ndarray:
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        r2 = X * X + Y * Y
        d2 = self.alpha * r2 + (1.0 - self.alpha) * Z * Z
        d = np.sqrt(np.maximum(d2, 1e-300))
        u = self.fx * X / d + self.cx
        v = self.fy * Y / d + self.cy
        return np.stack([u, v], axis=-1)

    def _unproject(self, uv: np.ndarray) -> np.ndarray:
        xd = (uv[..., 0] - self.cx) / self.fx
        yd = (uv[..., 1] - self.cy) / self.fy
        rho2 = xd * xd + yd * yd
        a = self.alpha
        d2 = (1.0 - a) / np.maximum(1.0 + (1.0 - 2.0 * a) * rho2, 1e-300)
        zp = np.sqrt(np.maximum(1.0 - rho2 * d2, 0.0))
        rays = np.stack([xd * np.sqrt(d2), yd * np.sqrt(d2), zp], axis=-1)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays / np.maximum(norms, 1e-12)

    def max_rho(self) -> float:
        return 1.0 / np.sqrt(max(self.alpha, 1e-9))

    def valid_mask(self, roundtrip_tol_px: float = 0.25) -> np.ndarray:
        uu, vv = self.pixel_grid()
        uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        rays = self._unproject(uv)
        finite = np.all(np.isfinite(rays), axis=-1)
        front = rays[:, 2] > 1e-9
        reproj = np.full_like(uv, np.nan)
        if np.any(finite & front):
            m = finite & front
            reproj[m] = self._project(rays[m])
        err = np.linalg.norm(reproj - uv, axis=-1)
        ok = finite & front & (err < roundtrip_tol_px)
        return ok.reshape(self.height, self.width)

    def fov_deg(self) -> float:
        mask = self.valid_mask()
        ys, xs = np.nonzero(mask)
        uv = np.stack([xs, ys], axis=-1).astype(np.float64)
        rays = self.unproject(uv)
        thetas = np.arccos(np.clip(rays[..., 2], -1, 1))
        return float(np.degrees(2.0 * thetas.max())) if len(thetas) else 0.0

    def intrinsics_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def distortion_dict(self) -> dict | None:
        return {"alpha": self.alpha}

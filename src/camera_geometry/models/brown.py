"""Brown-Conrady (plumb_bob) model for normal lenses.

x' = x(1 + k1 r^2 + k2 r^4 + k3 r^6) + [2 p1 x y + p2 (r^2 + 2 x^2)]
y' = y(1 + k1 r^2 + k2 r^4 + k3 r^6) + [p1 (r^2 + 2 y^2) + 2 p2 x y]
u = fx x' + cx ,  v = fy y' + cy

Unprojection via Newton on the 2x2 system (deterministic, ~10 iters).
Not a fisheye model: use for normal-FOV cameras only.
"""

from __future__ import annotations

import numpy as np

from .base import CameraModel

_NEWTON_ITERS = 20


class BrownConrady(CameraModel):
    model_type = "brown_conrady"

    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                 width: int = 0, height: int = 0):
        super().__init__(width or 640, height or 480)
        if min(fx, fy) <= 0:
            raise ValueError("fx/fy must be positive")
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.k1, self.k2, self.k3 = float(k1), float(k2), float(k3)
        self.p1, self.p2 = float(p1), float(p2)

    def _distort_normalized(self, x: np.ndarray, y: np.ndarray):
        r2 = x * x + y * y
        poly = 1.0 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
        xd = x * poly + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * poly + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return xd, yd

    def _project(self, xyz: np.ndarray) -> np.ndarray:
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = X / Z
            y = Y / Z
        xd, yd = self._distort_normalized(x, y)
        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        return np.stack([u, v], axis=-1)

    def _unproject(self, uv: np.ndarray) -> np.ndarray:
        ux = (uv[..., 0] - self.cx) / self.fx
        uy = (uv[..., 1] - self.cy) / self.fy
        x, y = ux.copy(), uy.copy()
        for _ in range(_NEWTON_ITERS):
            xd, yd = self._distort_normalized(x, y)
            ex, ey = xd - ux, yd - uy
            if float(np.max(np.abs(ex) + np.abs(ey), initial=0.0)) < 1e-12:
                break
            r2 = x * x + y * y
            poly = 1.0 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
            ddx_dx = poly + 2.0 * self.p1 * y + 6.0 * self.p2 * x
            ddx_dy = 2.0 * self.p1 * x + 4.0 * self.p2 * y
            ddy_dx = 4.0 * self.p1 * x + 2.0 * self.p2 * y
            ddy_dy = poly + 4.0 * self.p1 * y + 2.0 * self.p2 * x
            det = ddx_dx * ddy_dy - ddx_dy * ddy_dx
            safe = np.where(np.abs(det) > 1e-12, det, np.nan)
            jx = (ddy_dy * ex - ddx_dy * ey) / safe
            jy = (ddx_dx * ey - ddy_dx * ex) / safe
            x -= np.nan_to_num(jx, nan=0.0)
            y -= np.nan_to_num(jy, nan=0.0)
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)
        return rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)

    def valid_mask(self, tol_px: float = 0.25) -> np.ndarray:
        uu, vv = self.pixel_grid()
        uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        rays = self._unproject(uv)
        finite = np.all(np.isfinite(rays), axis=-1)
        reproj = np.full_like(uv, np.nan)
        if np.any(finite):
            reproj[finite] = self._project(rays[finite])
        err = np.linalg.norm(reproj - uv, axis=-1)
        ok = finite & (err < tol_px)
        return ok.reshape(self.height, self.width)

    def intrinsics_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def distortion_dict(self) -> dict | None:
        return {"k1": self.k1, "k2": self.k2, "k3": self.k3, "p1": self.p1, "p2": self.p2}

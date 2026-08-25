"""Double Sphere camera model (Usenko et al., 3DV 2018).

Params: fx, fy, cx, cy, xi, alpha (0 <= alpha <= 1, xi >= 0).
Projection of p=(x,y,z), z>0:
  d1 = ||p||
  X2 = p/d1 + xi*e_z            (first unit sphere, shifted)
  d2 = ||X2||
  denom = alpha*d2 + (1-alpha)*X2z
  u = fx*X2x/denom + cx ,  v = fy*X2y/denom + cy
xi=0, alpha=0 reduces to pinhole.
Unprojection solved in closed form (see derive notes in tests).
"""

from __future__ import annotations

import numpy as np

from .base import CameraModel


class DoubleSphere(CameraModel):
    model_type = "double_sphere"

    def __init__(self, fx, fy, cx, cy, xi: float, alpha: float, width: int, height: int):
        super().__init__(width, height)
        if min(fx, fy) <= 0:
            raise ValueError("fx/fy must be positive")
        if xi < 0:
            raise ValueError("xi must be >= 0")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.xi = float(xi)
        self.alpha = float(alpha)

    def _project(self, xyz: np.ndarray) -> np.ndarray:
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        d1 = np.sqrt(X * X + Y * Y + Z * Z)
        X2z = Z + self.xi * d1
        d2 = np.sqrt(X * X + Y * Y + X2z * X2z)
        denom = self.alpha * d2 + (1.0 - self.alpha) * X2z
        denom = np.where(np.abs(denom) < 1e-300, np.nan, denom)
        u = self.fx * X / denom + self.cx
        v = self.fy * Y / denom + self.cy
        return np.stack([u, v], axis=-1)

    def _solve_mz(self, rho2: np.ndarray) -> list[np.ndarray]:
        """All real candidate solutions of 1 = a*sqrt(rho2+mz^2) + (1-a)*mz.

        From squaring: (2a-1) mz^2 + 2(1-a) mz + (a^2 rho2 - 1) = 0.
        Handles a=0 (mz=1) and a=1 (mz=sqrt(1-rho2)) via the same formula.
        """
        a = self.alpha
        A = 2.0 * a - 1.0
        B = 2.0 * (1.0 - a)
        C = a * a * rho2 - 1.0
        if abs(A) < 1e-12:  # a = 0.5: linear
            return [np.full_like(rho2, -C / B)]
        disc = np.maximum(B * B - 4.0 * A * C, 0.0)
        root = np.sqrt(disc)
        return [(-B + root) / (2.0 * A), (-B - root) / (2.0 * A)]

    def _unproject(self, uv: np.ndarray) -> np.ndarray:
        mx = (uv[..., 0] - self.cx) / self.fx
        my = (uv[..., 1] - self.cy) / self.fy
        rho2 = mx * mx + my * my

        best = np.full(uv.shape[:-1] + (3,), np.nan)
        best_err = np.full(uv.shape[:-1], np.inf)
        for mz in self._solve_mz(rho2):
            s = rho2 + mz * mz
            valid0 = np.isfinite(mz) & (mz > 0) & (s > 1e-12)
            # D from ||D(m, mz) - xi ez|| = 1, positive branch
            disc = mz * mz + rho2 * (1.0 - self.xi * self.xi)
            valid = valid0 & (disc >= 0)
            D = (self.xi * np.where(valid, mz, 1.0) + np.sqrt(np.where(valid, disc, 1.0))) / np.where(valid, s, 1.0)
            z1 = D * np.where(valid, mz, 1.0) - self.xi
            valid &= D > 0
            valid &= z1 > 1e-9
            if not np.any(valid):
                continue
            rays = np.stack([D * mx, D * my, z1], axis=-1)
            rays /= np.linalg.norm(np.where(valid[:, None], rays, 1.0), axis=-1, keepdims=True)
            # numeric forward check to reject extraneous squared roots
            back = self._project(np.where(valid[:, None], rays, 1.0))
            err = np.linalg.norm(np.nan_to_num(back - uv), axis=-1)
            better = valid & (err < best_err)
            best_err = np.where(better, err, best_err)
            best[better] = rays[better]
        return best

    def valid_mask(self, roundtrip_tol_px: float = 0.25) -> np.ndarray:
        uu, vv = self.pixel_grid()
        uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        rays = self._unproject(uv)
        finite = np.all(np.isfinite(rays), axis=-1)
        front = rays[:, 2] > 1e-9
        reproj = np.full_like(uv, np.nan)
        m = finite & front
        if np.any(m):
            reproj[m] = self._project(rays[m])
        err = np.linalg.norm(reproj - uv, axis=-1)
        ok = finite & front & (err < roundtrip_tol_px)
        return ok.reshape(self.height, self.width)

    def intrinsics_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def distortion_dict(self) -> dict | None:
        return {"xi": self.xi, "alpha": self.alpha}

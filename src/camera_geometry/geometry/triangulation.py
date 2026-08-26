"""Native stereo geometry without rectification.

Two bearing rays + T_left_right -> midpoint triangulation.
"""

from __future__ import annotations

import numpy as np

from ..transforms import SE3


def triangulate_midpoint(
    rays_left: np.ndarray,
    rays_right: np.ndarray,
    T_left_right: SE3,
    min_angle_deg: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate pixel pairs given unit rays in each camera frame.

    Returns (points_left [N,3], angular_residuals_deg [N]).
    Points are expressed in the LEFT camera frame.
    Pairs whose parallax angle < min_angle_deg get residual = inf.
    """
    rays_left = np.atleast_2d(np.asarray(rays_left, dtype=np.float64))
    rays_right = np.atleast_2d(np.asarray(rays_right, dtype=np.float64))

    c_l = np.zeros(3)
    c_r = T_left_right.t.copy()  # right center in left frame: P_L = R P_R + t
    d_r_in_l = rays_right @ T_left_right.R.T  # directions right->left frame

    # Vectorized closed-form midpoint triangulation:
    # solve min |s*d1 - t*d2 - b|^2 via 2x2 normal equations per point.
    b = c_r - c_l
    d11 = np.sum(d_r_in_l * d_r_in_l, axis=-1)
    d22 = np.sum(rays_left * rays_left, axis=-1)
    d12 = -np.sum(rays_left * d_r_in_l, axis=-1)
    rhs1 = np.sum(d_r_in_l * b, axis=-1)   # -d2 . b
    rhs2 = np.sum(rays_left * b, axis=-1)  #  d1 . b
    det = d22 * d11 - d12 * d12
    safe_det = np.where(np.abs(det) > 1e-12, det, np.nan)
    s = (rhs2 * d11 + rhs1 * d12) / safe_det
    t = -(rhs1 * d22 + rhs2 * d12) / safe_det

    p1 = rays_left * s[:, None]
    p2 = c_r + d_r_in_l * t[:, None]
    mid = (p1 + p2) / 2.0

    cosang = np.clip(np.sum(rays_left * d_r_in_l, axis=-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    depths = np.linalg.norm(mid, axis=-1)
    line_err = np.linalg.norm(p1 - p2, axis=-1)

    points = mid
    residuals = np.where(
        (ang >= min_angle_deg) & np.isfinite(depths) & (depths > 1e-9),
        line_err / np.maximum(depths, 1e-9),
        np.inf,
    )
    return points, residuals

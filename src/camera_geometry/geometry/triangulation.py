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

    points = np.empty((len(rays_left), 3))
    residuals = np.full(len(rays_left), np.inf)

    for i, (d1, d2) in enumerate(zip(rays_left, d_r_in_l)):
        A = np.stack([d1, -d2], axis=-1)
        b = c_r - c_l
        try:
            st, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        p1 = st[0] * d1
        p2 = c_r + st[1] * d2
        mid = (p1 + p2) / 2.0
        points[i] = mid
        cosang = np.clip(d1 @ d2, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        if ang < min_angle_deg:
            continue
        residuals[i] = float(np.linalg.norm(p1 - p2) / max(1e-9, np.linalg.norm(mid)))
    return points, residuals

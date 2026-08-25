"""Native stereo geometry helpers built directly on rays (no rectification)."""

from __future__ import annotations

import numpy as np

from ..models.base import CameraModel
from ..transforms import SE3
from ..geometry.triangulation import triangulate_midpoint


def pixel_pairs_to_points(
    cam_left: CameraModel,
    cam_right: CameraModel,
    T_left_right: SE3,
    pixels_left: np.ndarray,
    pixels_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """pixels [N,2] per camera -> 3D points in left frame + relative residuals."""
    rays_l = np.asarray(cam_left.pixel_to_ray(pixels_left), dtype=np.float64)
    rays_r = np.asarray(cam_right.pixel_to_ray(pixels_right), dtype=np.float64)
    return triangulate_midpoint(rays_l, rays_r, T_left_right)

"""Remap LUT construction and application.

Direction (backward mapping, no per-frame geometry):
    virtual pixel -> virtual pinhole ray
                  -> [optional T_dst_src] -> source camera frame ray
                  -> source camera project -> source fisheye pixel

Runtime: image + cached grid -> cv2.remap / torch grid_sample.
"""

from __future__ import annotations

import numpy as np

from ..models.base import CameraModel
from ..transforms import SE3


def create_remap_grid(
    src: CameraModel,
    dst: CameraModel,
    T_dst_src: SE3 | None = None,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (map_abs float32 [H_dst,W_dst,2] in source pixels, valid bool [H,W])."""
    uu, vv = dst.pixel_grid()
    uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)
    rays = np.asarray(dst.unproject(uv), dtype=np.float64)
    if T_dst_src is not None:
        rays = T_dst_src.apply_direction(rays)
    suv = np.asarray(src.project(rays), dtype=np.float64)
    inside = src.contains(suv.reshape(-1, 2), margin=margin)
    finite = np.all(np.isfinite(suv), axis=-1)

    if hasattr(src, "valid_mask"):
        vmask = src.valid_mask()
        h, w = vmask.shape
        ui = np.clip(np.round(suv[:, 0]).astype(int), 0, w - 1)
        vi = np.clip(np.round(suv[:, 1]).astype(int), 0, h - 1)
        inside &= vmask[vi, ui]

    valid = inside & finite
    grid = suv.astype(np.float32).reshape(dst.height, dst.width, 2)
    return grid, valid.reshape(dst.height, dst.width)


def to_cv_maps(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return grid[..., 0].copy(), grid[..., 1].copy()


def to_torch_grid(grid: np.ndarray, src_width: int, src_height: int) -> np.ndarray:
    """Absolute-pixel grid -> normalized [-1,1] grid for F.grid_sample(align_corners=True)."""
    gx = grid[..., 0] / max(src_width - 1, 1) * 2.0 - 1.0
    gy = grid[..., 1] / max(src_height - 1, 1) * 2.0 - 1.0
    return np.stack([gx, gy], axis=-1).astype(np.float32)


def remap_coverage(valid: np.ndarray) -> float:
    return float(valid.mean())

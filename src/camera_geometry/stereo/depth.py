"""Rectified stereo depth: Z = f_rectified * B / d.

Never use the raw fisheye fx for rectified disparity depth.
"""

from __future__ import annotations

import cv2
import numpy as np


def disparity_to_depth(disparity_px, fx_rectified: float, baseline_m: float) -> np.ndarray:
    disp = np.asarray(disparity_px, dtype=np.float64)
    depth = np.full(disp.shape, np.inf)
    valid = np.isfinite(disp) & (disp > 0.5)
    depth[valid] = fx_rectified * baseline_m / disp[valid]
    return depth


def depth_to_disparity(depth_m, fx_rectified: float, baseline_m: float) -> np.ndarray:
    z = np.asarray(depth_m, dtype=np.float64)
    disp = np.full(z.shape, 0.0)
    valid = np.isfinite(z) & (z > 1e-6)
    disp[valid] = fx_rectified * baseline_m / z[valid]
    return disp


def compute_disparity_sgbm(
    img_left_rect: np.ndarray,
    img_right_rect: np.ndarray,
    num_disparities: int = 192,
    block_size: int = 7,
) -> np.ndarray:
    gray_l = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)
    sgbm = cv2.StereoSGBM_create(  # type: ignore[attr-defined]
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        uniquenessRatio=8,
        speckleWindowSize=120,
        speckleRange=2,
        disp12MaxDiff=1,
    )
    disp = sgbm.compute(gray_l, gray_r).astype(np.float64) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def stereo_depth(
    img_left_rect: np.ndarray,
    img_right_rect: np.ndarray,
    fx_rectified: float,
    baseline_m: float,
    **sgbm_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (disparity [px], depth [m])."""
    disp = compute_disparity_sgbm(img_left_rect, img_right_rect, **sgbm_kwargs)
    return disp, disparity_to_depth(disp, fx_rectified, baseline_m)

"""Fisheye stereo rectification into a shared virtual pinhole pair.

Both virtual cameras share identical intrinsics and orientation; the baseline
lies on the rectified x-axis => horizontal epipolar lines, Z = fx * B / d.

Virtual frame V basis expressed in left-camera frame: columns of R_rect_left.
d_V = R_rect_left @ d_left ;  d_V = R_rect_right @ d_right
R_rect_right = R_rect_left^T @ R_T_left_right
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import CameraModel
from ..models.pinhole import PinholeCamera
from ..transforms import SE3, camera_center_in_reference
from ..virtual_camera.remap import create_remap_grid


@dataclass
class StereoRectifier:
    virtual_camera: PinholeCamera
    T_left_right: SE3
    rotation_virtual_from_left: np.ndarray
    rotation_virtual_from_right: np.ndarray
    baseline_m: float
    grid_left: np.ndarray
    grid_right: np.ndarray
    valid_left: np.ndarray
    valid_right: np.ndarray
    fov_x_deg: float

    def disparity_to_depth(self, disparity_px):
        disp = np.asarray(disparity_px, dtype=np.float64)
        depth = np.full_like(disp, np.inf, dtype=np.float64)
        m = disp > 0.5
        depth[m] = self.virtual_camera.fx * self.baseline_m / disp[m]
        return depth


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def build_stereo_rectifier(
    cam_left: CameraModel,
    cam_right: CameraModel,
    T_left_right: SE3,
    width: int,
    height: int,
    fov_x_deg: float = 100.0,
) -> StereoRectifier:
    c_r = camera_center_in_reference(T_left_right)
    baseline = float(np.linalg.norm(c_r))
    if baseline < 1e-9:
        raise ValueError("degenerate baseline (~0)")
    e1 = _normalize(c_r)  # x axis: left center -> right center

    z_axis = np.array([0.0, 0.0, 1.0])
    z_l = z_axis
    z_r = T_left_right.R @ z_axis
    z_avg = _normalize(z_l + z_r)

    forward = z_avg - np.dot(z_avg, e1) * e1
    if np.linalg.norm(forward) < 1e-6:
        forward = z_l - np.dot(z_l, e1) * e1
    c3 = _normalize(forward)
    c1 = e1
    c2 = np.cross(c3, c1)

    R_rect_left = np.column_stack([c1, c2, c3])
    R_rect_right = R_rect_left.T @ T_left_right.R

    virtual = PinholeCamera.from_fov(width, height, fov_x_deg)
    T_v_left = SE3(R_rect_left, np.zeros(3))
    T_v_right = SE3(R_rect_right, np.zeros(3))
    grid_left, valid_left = create_remap_grid(cam_left, virtual, T_v_left)
    grid_right, valid_right = create_remap_grid(cam_right, virtual, T_v_right)

    return StereoRectifier(
        virtual_camera=virtual,
        T_left_right=T_left_right,
        rotation_virtual_from_left=R_rect_left,
        rotation_virtual_from_right=R_rect_right,
        baseline_m=baseline,
        grid_left=grid_left,
        grid_right=grid_right,
        valid_left=valid_left,
        valid_right=valid_right,
        fov_x_deg=fov_x_deg,
    )

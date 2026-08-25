"""Rolling shutter support (spec section 28).

Lens distortion and rolling shutter are strictly separated: KB parameters
never fit motion distortion. Given
    t(y) = t0 + (y / H) * T_readout
and a per-timestamp pose provider (e.g. IMU/VIO interpolation), this module
produces row-aware rays and a row-aware undistortion map.

MVP scope: the row-time model, row-aware pixel_to_ray, and the pose-hook
contract. The IMU-integrating pose provider itself belongs to the VIO stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .models.base import CameraModel
from .transforms import SE3


@dataclass(frozen=True)
class RollingShutterModel:
    width: int
    height: int
    readout_time_ns: float
    frame_ts_ns: float

    def row_time_ns(self, v: np.ndarray | float) -> np.ndarray | float:
        """t(y) = t0 + (y / H) * T_readout  (y = row index, 0 at top)."""
        return self.frame_ts_ns + (np.asarray(v, dtype=np.float64) / max(self.height - 1, 1)) * self.readout_time_ns

    def pixel_times_ns(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        return np.atleast_1d(self.row_time_ns(uv[..., 1]))


PoseProvider = Callable[[float], SE3]
"""Contract: pose(t_ns) -> T_world_cam at that instant (IMU/VIO interpolation)."""


class RowAwareCamera:
    """Wraps a CameraModel with the rolling-shutter time model.

    pixel_to_ray_rs(uv, pose_provider) returns, for every pixel, the unit ray
    in the WORLD frame at that pixel's own exposure time:
        ray_w = R_wc(t(y)) @ cam.unproject(uv)
    Geometry consumers (SLAM/VIO) should use this instead of a single
    per-frame pose when readout is significant relative to motion.
    """

    def __init__(self, camera: CameraModel, rs: RollingShutterModel):
        self.camera = camera
        self.rs = rs

    def pixel_times_ns(self, uv: np.ndarray) -> np.ndarray:
        return self.rs.pixel_times_ns(uv)

    def pixel_to_ray_rs(self, uv: np.ndarray, pose_provider: PoseProvider) -> tuple[np.ndarray, np.ndarray]:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        times = np.atleast_1d(self.rs.pixel_times_ns(uv))
        rays_cam = np.asarray(self.camera.unproject(uv))
        rays_w = np.empty_like(rays_cam)
        for i, t in enumerate(times):
            T_wc = pose_provider(float(t))
            rays_w[i] = T_wc.apply_direction(rays_cam[i])
        return rays_w, times

    def row_undistort_map(
        self,
        virtual_camera: CameraModel,
        pose_provider: PoseProvider,
        t0_ns: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward remap grid that undistorts AND de-rolls: every virtual
        pixel's source ray is computed at that ROW's exposure time.

        For each target row v_dst the virtual camera is assumed exposed at
        t(v_dst); source rays are rotated by the RELATIVE rotation between
        the row time and the frame reference time (translation over readout
        is a second-order effect for ego motion at typical readout times and
        is ignored in this MVP).
        """
        t0 = self.rs.frame_ts_ns if t0_ns is None else t0_ns
        uu, vv = virtual_camera.pixel_grid()
        uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        rays_v = np.asarray(virtual_camera.unproject(uv))

        rows = vv.reshape(-1).astype(int)
        row_times = np.asarray(self.rs.row_time_ns(rows))
        rel_rots = np.empty((len(rows), 3, 3))
        ref_R = pose_provider(float(t0)).R
        for k, t in enumerate(np.unique(row_times)):
            sel = row_times == t
            rel_rots[sel] = pose_provider(float(t)).R.T @ ref_R

        rays_cam = np.einsum("kij,kj->ki", rel_rots, rays_v)
        suv = np.asarray(self.camera.project(rays_cam))
        grid = suv.astype(np.float32).reshape(virtual_camera.height, virtual_camera.width, 2)
        valid = np.all(np.isfinite(grid), axis=-1) & self.camera.contains(grid.reshape(-1, 2)).reshape(
            virtual_camera.height, virtual_camera.width
        )
        return grid, valid

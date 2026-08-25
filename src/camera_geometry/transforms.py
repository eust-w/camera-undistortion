"""Rigid body transforms with explicit T_A_B naming convention.

T_A_B maps geometry from frame B into frame A:  P_A = R @ P_B + t.
Directions (rays) transform as:                  d_A = R @ d_B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def as_rotation(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {R.shape}")
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det_err = abs(float(np.linalg.det(R)) - 1.0)
    if ortho_err > 1e-3 or det_err > 1e-3:
        raise ValueError(f"invalid rotation: ||R^TR-I||={ortho_err:.2e} |det(R)-1|={det_err:.2e}")
    return R


@dataclass(frozen=True)
class SE3:
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def identity() -> "SE3":
        return SE3(np.eye(3), np.zeros(3))

    @staticmethod
    def from_Rt(R, t) -> "SE3":
        return SE3(as_rotation(R), np.asarray(t, dtype=np.float64).reshape(3))

    @staticmethod
    def from_matrix(T) -> "SE3":
        T = np.asarray(T, dtype=np.float64)
        return SE3.from_Rt(T[:3, :3], T[:3, 3])

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def apply(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return p @ self.R.T + self.t

    def apply_direction(self, dirs) -> np.ndarray:
        d = np.asarray(dirs, dtype=np.float64)
        return d @ self.R.T

    def inverse(self) -> "SE3":
        R_inv = self.R.T
        return SE3(R_inv, -R_inv @ self.t)

    def compose(self, other: "SE3") -> "SE3":
        """self.compose(other) == self * other == T_A_C where self=T_A_B, other=T_B_C."""
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def __matmul__(self, other: "SE3") -> "SE3":
        return self.compose(other)


def rotx(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def roty(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotz(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def camera_center_in_reference(T_reference_self: SE3) -> np.ndarray:
    """Position of 'self' camera center expressed in the reference frame.

    With T_A_B: P_A = R @ P_B + t, setting P_B = 0 gives center = t.
    """
    return T_reference_self.t


def transform_points_to_self(T_reference_self: SE3, points_reference) -> np.ndarray:
    """Reference-frame points -> self-camera frame (inverse transform)."""
    return T_reference_self.inverse().apply(points_reference)

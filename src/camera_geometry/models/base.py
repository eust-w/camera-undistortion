"""Unified camera geometry API.

Every downstream consumer must go through this interface:
    camera.project(xyz)        ray/point -> pixel
    camera.unproject(uv)       pixel -> unit bearing ray
    camera.pixel_to_ray(uv)    alias of unproject
    camera.ray_to_pixel(ray)   alias of project
    camera.valid_mask()        [H,W] bool, pixels with numerically valid rays

Core math is NumPy. Inputs may be np.ndarray or torch.Tensor (CPU);
when a torch tensor is passed in, a torch tensor is returned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _load_torch():
    try:
        import torch  # noqa: F401

        return torch
    except ImportError:
        return None


TORCH_AVAILABLE = _load_torch() is not None


def to_numpy(x) -> np.ndarray:
    if TORCH_AVAILABLE:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return np.asarray(x)


class CameraModel(ABC):
    """Canonical representation: pixel <-> unit ray."""

    model_type = "camera"

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

    # ---- public API (framework passthrough) -------------------------
    def project(self, xyz):
        arr = to_numpy(xyz)
        return self._maybe_torch(xyz, self._project(arr))

    def unproject(self, uv):
        arr = to_numpy(uv)
        return self._maybe_torch(uv, self._unproject(arr))

    def pixel_to_ray(self, uv):
        return self.unproject(uv)

    def ray_to_pixel(self, ray):
        return self.project(ray)

    # ---- abstract core ----------------------------------------------
    @abstractmethod
    def _project(self, xyz: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _unproject(self, uv: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def valid_mask(self) -> np.ndarray:
        """[H,W] bool mask of pixels whose unprojection is numerically valid."""

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _maybe_torch(input_like, result: np.ndarray):
        if TORCH_AVAILABLE:
            import torch

            if isinstance(input_like, torch.Tensor):
                return torch.from_numpy(np.ascontiguousarray(result))
        return result

    def pixel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(self.width, dtype=np.float64)
        ys = np.arange(self.height, dtype=np.float64)
        return np.meshgrid(xs, ys)

    def contains(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        u, v = uv[..., 0], uv[..., 1]
        return (
            (u >= -margin)
            & (u <= self.width - 1 + margin)
            & (v >= -margin)
            & (v <= self.height - 1 + margin)
        )

    def intrinsics_dict(self) -> dict:
        raise NotImplementedError

    def distortion_dict(self) -> dict | None:
        return None

    def summary(self) -> dict:
        return {
            "model": self.model_type,
            "resolution": [self.width, self.height],
            "intrinsics": self.intrinsics_dict(),
            "distortion": self.distortion_dict(),
        }

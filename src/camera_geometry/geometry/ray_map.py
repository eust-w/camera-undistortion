"""Ray map / valid mask generation and caching.

For a fixed calibration, ray_map[H,W,3] holds the unit camera ray of every
pixel; valid_mask[H,W] excludes pixels without numerically valid rays.
All geometry downstream must use ray_map + valid_mask.
"""

from __future__ import annotations

import numpy as np

from ..models.base import CameraModel


def generate_ray_map(camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """Return (rays float64 [H,W,3], valid bool [H,W])."""
    uu, vv = camera.pixel_grid()
    uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)
    rays = np.asarray(camera.unproject(uv), dtype=np.float64)
    if not isinstance(rays, np.ndarray):  # torch passthrough guard
        import numpy as _np

        rays = _np.asarray(rays)
    rays = rays.reshape(camera.height, camera.width, 3)
    valid = camera.valid_mask()
    return rays, valid


def save_ray_map(path, rays: np.ndarray, valid: np.ndarray, fp16: bool = True) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rays.astype(np.float16) if fp16 else rays.astype(np.float32)
    np.savez(path, rays=data, valid=valid.astype(np.bool_), format_version=1)


def load_ray_map(path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["rays"].astype(np.float32), z["valid"].astype(bool)


def sample_valid_pixels(valid: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Random valid pixel coordinates [n,2] (u,v), float64."""
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(valid)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float64)

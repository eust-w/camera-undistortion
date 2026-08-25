"""PyTorch GPU runtime backend (batched grid_sample).

Batch: [B,C,H_src,W_src] -> grid_sample -> [B,C,H_dst,W_dst]
Grids are precomputed once per calibration and uploaded once.
"""

from __future__ import annotations

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:  # pragma: no cover
    TORCH_OK = False

from ..virtual_camera.remap import to_torch_grid


def _require_torch():
    if not TORCH_OK:
        raise RuntimeError("torch is not installed; pip install 'auto-camera-geometry[torch]'")


def make_sampling_grid(grid_abs: np.ndarray, src_width: int, src_height: int, device=None, dtype=None):
    _require_torch()
    dtype = dtype or torch.float32
    g = torch.from_numpy(to_torch_grid(grid_abs, src_width, src_height)).to(dtype=dtype)
    if device is not None:
        g = g.to(device)
    return g


def remap_batch(images: "torch.Tensor", sampling_grid: "torch.Tensor") -> "torch.Tensor":
    """images [B,C,H,W], sampling_grid [H_dst,W_dst,2] normalized."""
    _require_torch()
    grid = sampling_grid.unsqueeze(0).expand(images.shape[0], -1, -1, -1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

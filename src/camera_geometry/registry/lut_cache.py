"""LUT cache: all fixed geometry artifacts cached by content key.

cache_key = hash(calibration_id + output_projection + output_resolution + output_fov)
Artifacts: ray_map / valid_mask / virtual grids / stereo grids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def make_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


class LUTCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str, name: str) -> Path:
        return self.root / f"{key}__{name}.npz"

    def has(self, key: str, name: str) -> bool:
        return self.path(key, name).exists()

    def get(self, key: str, name: str) -> dict[str, np.ndarray] | None:
        p = self.path(key, name)
        if not p.exists():
            return None
        with np.load(p) as z:
            return {k: z[k] for k in z.files if not k.startswith("_")}

    def put(self, key: str, name: str, arrays: dict[str, np.ndarray], meta: dict | None = None) -> Path:
        p = self.path(key, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays, _meta=json.dumps(meta or {}))
        tmp.replace(p)
        return p

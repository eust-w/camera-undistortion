"""Golden regression tests (spec section 44).

Skips when no golden data exists. Generate with:
    python scripts/export_golden.py --source synthetic --tag synth_default
Vendor truth (Phase 0):
    python scripts/export_golden.py --source orbbec_sdk --tag orbbec_default
    (requires implementing orbbec_pixel_to_ray in the export script)
"""

from pathlib import Path

import numpy as np
import pytest

from camera_geometry.calibration.parser import load_device_calibration
from camera_geometry.models.factory import build_camera

GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "tests" / "golden"
TAGS = sorted(p.name for p in GOLDEN_ROOT.iterdir() if (p / "pixel_to_ray.npz").exists()) if GOLDEN_ROOT.exists() else []


@pytest.mark.skipif(not TAGS, reason="no golden data; run scripts/export_golden.py")
@pytest.mark.parametrize("tag", TAGS)
def test_golden_pixel_to_ray_regression(tag):
    data = np.load(GOLDEN_ROOT / tag / "pixel_to_ray.npz")
    pixels, rays_ref = data["pixels"], data["rays"]
    cfg = load_device_calibration(GOLDEN_ROOT / tag / "config.yaml")
    cam = build_camera(cfg.camera("IR_L"))
    rays_now = np.asarray(cam.unproject(pixels))
    ang = np.degrees(np.arccos(np.clip(np.sum(rays_now * rays_ref, axis=-1), -1, 1)))
    # regression gate: median must stay at machine precision for synthetic
    # golden; vendor golden uses 0.05 deg (Phase-0 projection target)
    gate = 0.05 if str(data.get("source")) == "orbbec_sdk" else 1e-6
    assert float(np.median(ang)) < gate, f"golden regression: median {np.median(ang)} deg"


@pytest.mark.skipif(not TAGS, reason="no golden data")
@pytest.mark.parametrize("tag", TAGS)
def test_golden_ray_to_pixel_regression(tag):
    data = np.load(GOLDEN_ROOT / tag / "ray_to_pixel.npz")
    rays, pixels_ref = data["rays"], data["pixels"]
    cfg = load_device_calibration(GOLDEN_ROOT / tag / "config.yaml")
    cam = build_camera(cfg.camera("IR_L"))
    pixels_now = np.asarray(cam.project(rays))
    err = np.linalg.norm(pixels_now - pixels_ref, axis=-1)
    gate = 0.05 if str(data.get("source")) == "orbbec_sdk" else 1e-6
    assert float(np.percentile(err, 99)) < gate

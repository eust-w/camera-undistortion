import numpy as np
import pytest

from camera_geometry.models.double_sphere import DoubleSphere
from camera_geometry.models.eucm import EUCM
from camera_geometry.models.pinhole import PinholeCamera
from camera_geometry.calibration.parser import load_device_calibration
from camera_geometry.models.factory import build_camera

W, H = 640, 480


def rays_in_fov(rng, n, max_deg):
    out = []
    while len(out) < n:
        r = rng.normal(size=(n, 3))
        r[:, 2] = np.abs(r[:, 2]) + 1e-6
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        th = np.degrees(np.arccos(np.clip(r[:, 2], -1, 1)))
        out += list(r[th < max_deg])
    return np.array(out[:n])


def roundtrip_stats(cam, max_deg, seed=0):
    rng = np.random.default_rng(seed)
    rays = rays_in_fov(rng, 15000, max_deg)
    uv = cam.project(rays)
    ok = np.all(np.isfinite(uv), axis=-1)
    back = cam.unproject(uv[ok])
    ang = np.degrees(np.arccos(np.clip(np.sum(back * rays[ok], axis=-1), -1, 1)))
    return ang, ok


def test_eucm_alpha0_is_pinhole():
    e = EUCM(500, 505, 320, 240, 0.0, W, H)
    p = PinholeCamera(500, 505, 320, 240, W, H)
    rays = rays_in_fov(np.random.default_rng(1), 5000, 60)
    assert np.abs(e.project(rays) - p.project(rays)).max() < 1e-9
    uv = p.project(rays)
    assert np.abs(e.unproject(uv) - p.unproject(uv)).max() < 1e-12


def test_eucm_roundtrip():
    cam = EUCM(500, 505, 320, 240, 0.7, W, H)
    ang, ok = roundtrip_stats(cam, 45)
    assert ok.sum() > 14000
    assert np.percentile(ang[ok], 99) < 1e-3  # degrees


def test_eucm_alpha_must_be_below_one():
    with pytest.raises(ValueError):
        EUCM(500, 500, 320, 240, 1.0, W, H)


def test_eucm_valid_mask_excludes_beyond_fov():
    # corner rho = hypot(319.5/500, 239.5/500) = 0.80 < 1/sqrt(0.8) -> fully valid
    cam = EUCM(500, 500, 320, 240, 0.8, W, H)
    assert cam.valid_mask().mean() == 1.0
    # fx=300, alpha=0.9: corner rho = 1.33 > 1/sqrt(0.9) = 1.054 -> corners invalid
    cam2 = EUCM(300, 300, 320, 240, 0.9, W, H)
    m2 = cam2.valid_mask()
    assert m2.mean() < 0.95
    assert m2[int(240), int(320)] and not m2[0, 0]


def test_ds_xi0_alpha0_is_pinhole():
    d = DoubleSphere(500, 505, 320, 240, 0.0, 0.0, W, H)
    p = PinholeCamera(500, 505, 320, 240, W, H)
    rays = rays_in_fov(np.random.default_rng(2), 5000, 60)
    assert np.abs(d.project(rays) - p.project(rays)).max() < 1e-9


@pytest.mark.parametrize("xi,alpha,max_deg", [(0.8, 0.6, 35), (0.2, 0.9, 40), (1.2, 0.7, 45), (0.5, 0.5, 45)])
def test_ds_roundtrip(xi, alpha, max_deg):
    cam = DoubleSphere(500, 505, 320, 240, xi, alpha, W, H)
    ang, ok = roundtrip_stats(cam, max_deg)
    assert ok.sum() > 10000
    assert np.percentile(ang[ok], 99) < 1e-3


def test_ds_valid_mask():
    cam = DoubleSphere(500, 505, 320, 240, 0.8, 0.6, W, H)
    assert cam.valid_mask().mean() == 1.0  # corner rho 0.80 < 1
    cam2 = DoubleSphere(300, 300, 320, 240, 0.8, 0.6, W, H)  # corner rho 1.33 > 1
    m2 = cam2.valid_mask()
    assert m2.mean() < 0.95
    assert m2[int(240), int(320)] and not m2[0, 0]


# ----------------------------------------------------------- schema/factory
def test_schema_and_factory_support_new_models(tmp_path):
    cfg = tmp_path / "d.yaml"
    cfg.write_text(
        """
device: {serial_number: S1}
cameras:
  - id: EU
    resolution: {width: 640, height: 480}
    model: {type: eucm, params: {alpha: 0.7}}
    intrinsics: {fx: 500, fy: 505, cx: 320, cy: 240}
  - id: DS
    resolution: {width: 640, height: 480}
    model: {type: ds, params: {xi: 0.8, alpha: 0.6}}
    intrinsics: {fx: 500, fy: 505, cx: 320, cy: 240}
"""
    )
    calib = load_device_calibration(cfg)
    e = build_camera(calib.camera("EU"))
    d = build_camera(calib.camera("DS"))
    assert isinstance(e, EUCM) and abs(e.alpha - 0.7) < 1e-12
    assert isinstance(d, DoubleSphere) and abs(d.xi - 0.8) < 1e-12
    uv = np.array([[320.0, 240.0]])
    assert np.allclose(e.unproject(uv), [[0, 0, 1]], atol=1e-9)

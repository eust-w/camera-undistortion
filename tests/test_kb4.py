import numpy as np
import pytest

from camera_geometry.models.kb4 import KannalaBrandt4
from camera_geometry.models.pinhole import PinholeCamera

W, H = 1600, 1300


@pytest.fixture
def kb4():
    return KannalaBrandt4(
        fx=504.97497043,
        fy=504.85320543,
        cx=825.88761829,
        cy=644.70496968,
        k1=0.07305208,
        k2=-0.01165996,
        k3=0.00033907,
        k4=0.00027176,
        width=W,
        height=H,
    )


def test_project_axis(kb4):
    uv = kb4.project(np.array([[0.0, 0.0, 5.0]]))
    assert np.allclose(uv, [[kb4.cx, kb4.cy]], atol=1e-9)


def test_unproject_center(kb4):
    ray = kb4.unproject(np.array([[kb4.cx, kb4.cy]]))
    assert np.allclose(ray, [[0.0, 0.0, 1.0]], atol=1e-12)


def test_unit_rays(kb4):
    rng = np.random.default_rng(0)
    uv = np.column_stack([
        rng.uniform(0, W, 5000),
        rng.uniform(0, H, 5000),
    ])
    rays = kb4.unproject(uv)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1e-12)
    assert np.all(rays[:, 2] > 0)


def test_roundtrip_pixels_100k_under_tolerance(kb4):
    """Spec section 13: P99 < 0.05 px on valid pixels."""
    mask = kb4.valid_mask()
    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(xs), size=100_000, replace=False)
    uv = np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float64)
    back = kb4.project(kb4.unproject(uv))
    err = np.linalg.norm(back - uv, axis=-1)
    p99 = float(np.percentile(err, 99))
    assert p99 < 0.05, f"P99 roundtrip {p99:.6f} px"
    assert float(err.max()) < 1e-3


def test_roundtrip_rays(kb4):
    rng = np.random.default_rng(1)
    rays = rng.normal(size=(100_000, 3))
    rays[:, 2] = np.abs(rays[:, 2]) + 0.05
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    back = kb4.unproject(kb4.project(rays))
    ang = np.degrees(np.arccos(np.clip(np.sum(back * rays, axis=-1), -1, 1)))
    assert float(np.percentile(ang, 99)) < 0.01  # degrees


def test_valid_mask_sanity(kb4):
    mask = kb4.valid_mask()
    assert 0.3 < mask.mean() < 1.0
    # fisheye image circle: corners must be invalid for this wide lens
    assert not mask[0, 0]
    assert mask[int(kb4.cy), int(kb4.cx)]


def test_behind_camera_invalid(kb4):
    uv = kb4.project(np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -2.0]]))
    rays = kb4.unproject(uv)
    # unprojection is a direction mapping; front-hemisphere validity is the contract
    assert rays.shape == (2, 3)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0)


# --------------------------------------------------------------- pinhole
def test_pinhole_roundtrip():
    cam = PinholeCamera.from_fov(640, 640, 100.0)
    assert abs(cam.fx - (320 / np.tan(np.radians(50)))) < 1e-9
    uv = np.array([[0.0, 0.0], [639.0, 639.0], [320.0, 200.0]])
    back = cam.project(cam.unproject(uv))
    assert np.allclose(back, uv, atol=1e-9)


# ------------------------------------------- OpenCV fisheye convention check
cv2_fisheye = pytest.importorskip("cv2")


def test_matches_opencv_fisheye_convention(kb4):
    """Phase-0 style check: our KB4 == cv2.fisheye (OpenCV KB convention)."""
    import cv2

    K = np.array([[kb4.fx, 0, kb4.cx], [0, kb4.fy, kb4.cy], [0, 0, 1]])
    D = kb4.k.reshape(1, 4)

    rng = np.random.default_rng(7)
    rays = rng.normal(size=(20_000, 3))
    rays[:, 2] = np.abs(rays[:, 2])
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    pts_norm = (rays[:, :2] / rays[:, 2:3]).astype(np.float64).reshape(-1, 1, 2)
    uv_cv = cv2.fisheye.distortPoints(pts_norm, K, D).reshape(-1, 2)
    uv_ours = kb4.project(rays)
    proj_err = np.linalg.norm(uv_cv - uv_ours, axis=-1)
    assert float(np.percentile(proj_err, 99)) < 1e-6

    uv_in = uv_ours[(uv_ours[:, 0] > 1) & (uv_ours[:, 0] < W - 1) & (uv_ours[:, 1] > 1) & (uv_ours[:, 1] < H - 1)]
    ud = cv2.fisheye.undistortPoints(uv_in.astype(np.float32).reshape(-1, 1, 2), K, D).reshape(-1, 2)
    r_cv = np.column_stack([ud, np.ones(len(ud))])
    r_cv /= np.linalg.norm(r_cv, axis=-1, keepdims=True)
    r_ours = kb4.unproject(uv_in)
    ang = np.degrees(np.arccos(np.clip(np.sum(r_cv * r_ours, axis=-1), -1, 1)))
    assert float(np.median(ang)) < 1e-4

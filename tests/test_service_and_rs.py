"""Tests for the service API, metrics export, and rolling shutter model."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from camera_geometry.metrics import to_prometheus
from camera_geometry.models.kb4 import KannalaBrandt4
from camera_geometry.rolling_shutter import RollingShutterModel, RowAwareCamera
from camera_geometry.transforms import SE3
from camera_geometry.service import app

W, H = 1600, 1300


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    import os

    os.environ["CAMERA_GEOMETRY_HOME"] = str(home)
    yield TestClient(app)
    os.environ.pop("CAMERA_GEOMETRY_HOME", None)


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    cfg = tmp_path_factory.mktemp("cfg") / "device.yaml"
    cfg.write_text(
        """
device: {vendor: orbbec, serial_number: SVCTEST001}
cameras:
  - id: IR_L
    resolution: {width: 1600, height: 1300}
    model: {type: kannala_brandt_4}
    intrinsics: {fx: 504.97, fy: 504.85, cx: 825.88, cy: 644.70}
    distortion: {k1: 0.073, k2: -0.0117, k3: 0.00034, k4: 0.00027}
  - id: IR_R
    resolution: {width: 1600, height: 1300}
    model: {type: kannala_brandt_4}
    intrinsics: {fx: 505.10, fy: 505.00, cx: 826.10, cy: 645.20}
    distortion: {k1: 0.0731, k2: -0.0116, k3: 0.00035, k4: 0.00028}
    extrinsics:
      reference_camera: IR_L
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation_m: [0.12086, 0, 0]
calibration: {source: factory, version: v1}
"""
    )
    return str(cfg)


# ------------------------------------------------------------- rolling shutter
def test_row_time_model():
    rs = RollingShutterModel(W, H, readout_time_ns=10_000_000, frame_ts_ns=1_000_000_000)
    assert rs.row_time_ns(0) == 1_000_000_000
    assert rs.row_time_ns(H - 1) == 1_010_000_000
    mid = float(np.asarray(rs.row_time_ns(np.array([(H - 1) / 2])))[0])
    assert mid == 1_005_000_000


def test_row_aware_rays_use_per_row_pose():
    cam = KannalaBrandt4(505, 505, 800, 650, 0.072, -0.0117, 0.00034, 0.00027, W, H)
    rs = RollingShutterModel(W, H, readout_time_ns=20_000_000, frame_ts_ns=0.0)

    def pose(t_ns):
        # 90 deg/s rotation about y: 20 ms readout -> ~1.8 deg sweep
        ang = np.radians(90.0) * (t_ns / 1e9)
        return SE3.from_Rt(SE3.from_Rt.__wrapped__ if False else _roty(ang).R, [0, 0, 0]) if False else SE3.from_Rt(_roty(ang), [0, 0, 0])


    def _roty(a):
        from camera_geometry.transforms import roty as _r

        return _r(a)

    rac = RowAwareCamera(cam, rs)
    uv = np.array([[800.0, 0.0], [800.0, H - 1.0]])
    rays, times = rac.pixel_to_ray_rs(uv, pose)
    assert times[0] == 0.0 and times[1] == 20_000_000
    # same pixel column, different rows -> different world rays under rotation
    assert np.degrees(np.arccos(np.clip(rays[0] @ rays[1], -1, 1))) > 1.0
    # static pose must reduce to plain pixel_to_ray in world == cam frame
    rays_static, _ = rac.pixel_to_ray_rs(uv, lambda t: SE3.identity())
    assert np.allclose(rays_static, np.asarray(cam.pixel_to_ray(uv)), atol=1e-9)


def test_row_undistort_map_shapes():
    from camera_geometry.models.pinhole import PinholeCamera
    from camera_geometry.rolling_shutter import RowAwareCamera

    cam = KannalaBrandt4(505, 505, 800, 650, 0.072, -0.0117, 0.00034, 0.00027, W, H)
    rs = RollingShutterModel(W, H, readout_time_ns=5_000_000, frame_ts_ns=0.0)
    rac = RowAwareCamera(cam, rs)
    vcam = PinholeCamera.from_fov(320, 320, 90.0)
    grid, valid = rac.row_undistort_map(vcam, lambda t: SE3.identity())
    assert grid.shape == (320, 320, 2) and valid.shape == (320, 320)
    assert valid.mean() > 0.5
    # static pose: center of virtual view maps near the principal point
    c = grid[160, 160]
    assert abs(c[0] - 800) < 5 and abs(c[1] - 650) < 5


# ------------------------------------------------------------------ metrics
def test_prometheus_export():
    text = to_prometheus({
        "calibration_health_score": 0.96,
        "runtime_ms": {"lut": 120, "qa": 300},
        "calibration_id": "S1:IR_L:1600x1300:v1",
    })
    assert "calibration_health_score 0.96" in text
    assert "runtime_ms_lut 120" in text
    assert 'calibration_id{value="S1:IR_L:1600x1300:v1"} 1' in text


# ------------------------------------------------------------------- service
def test_service_resolve_and_get(client, config):
    r = client.post("/camera/resolve", json={"config_path": config, "camera_id": "IR_L"})
    assert r.status_code == 200
    body = r.json()
    assert body["calibration_id"] == "SVCTEST001:IR_L:1600x1300:v1"
    assert body["camera_model"] == "kannala_brandt_4"
    assert {"calibration_id", "camera_model", "quality_status", "artifact_uri"} <= set(body)

    r2 = client.get("/camera/SVCTEST001:IR_L:1600x1300:v1")
    assert r2.status_code == 404  # not registered until validated


def test_service_validate_then_get(client, config):
    r = client.post("/camera/validate", json={"config_path": config, "roundtrip_samples": 5000})
    assert r.status_code == 200
    assert r.json()["status"] in ("PASS", "DEGRADED")

    r2 = client.get("/camera/SVCTEST001:IR_L:1600x1300:v1")
    assert r2.status_code == 200
    assert r2.json()["quality_status"] in ("PASS", "DEGRADED")


def test_service_raymap_virtual(client, config):
    r = client.post("/camera/ray-map", json={"config_path": config, "camera_id": "IR_L"})
    assert r.status_code == 200
    assert 0 < r.json()["valid_pixel_ratio"] <= 1.0
    # second call must hit the cache (same artifact uri)
    r2 = client.post("/camera/ray-map", json={"config_path": config, "camera_id": "IR_L"})
    assert r2.json()["artifact_uri"] == r.json()["artifact_uri"]

    v = client.post("/camera/virtual-camera", json={"config_path": config, "camera_id": "IR_L"})
    assert v.status_code == 200
    assert v.json()["coverage"] > 0.9


def test_service_metrics_and_errors(client, config):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "camera_geometry_info" in r.text

    bad = client.post("/camera/resolve", json={"config_path": "/nonexistent.yaml", "camera_id": "IR_L"})
    assert bad.status_code == 400

    missing = client.get("/camera/UNKNOWN:ID")
    assert missing.status_code == 404

    h = client.get("/healthz")
    assert h.json()["status"] == "ok"

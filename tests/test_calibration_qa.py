
import numpy as np
import pytest

from camera_geometry.calibration.parser import load_device_calibration
from camera_geometry.calibration.resolver import CalibrationNotFound, CalibrationResolver
from camera_geometry.calibration.store import CalibrationRecord, CalibrationStore
from camera_geometry.calibration.validator import validate_calibration
from camera_geometry.registry.lut_cache import LUTCache, make_key
from camera_geometry.schema import DeviceCalibration


@pytest.fixture
def config_path(tmp_path):
    cfg = tmp_path / "device.yaml"
    cfg.write_text(
        """
schema_version: camera-geometry/v1
device: {vendor: orbbec, serial_number: TESTSERIAL1}
cameras:
  - id: IR_L
    resolution: {width: 1600, height: 1300}
    model: {type: kannala_brandt_4, convention: orbbec_kb_v1}
    intrinsics: {fx: 504.97, fy: 504.85, cx: 825.88, cy: 644.70}
    distortion: {k1: 0.07305208, k2: -0.01165996, k3: 0.00033907, k4: 0.00027176}
  - id: IR_R
    resolution: {width: 1600, height: 1300}
    model: {type: kannala_brandt_4}
    intrinsics: {fx: 505.10, fy: 505.00, cx: 826.10, cy: 645.20}
    distortion: {k1: 0.0731, k2: -0.0116, k3: 0.00035, k4: 0.00028}
    extrinsics:
      reference_camera: IR_L
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation_m: [0.12086, 0, 0]
shutter: {type: unknown, readout_time_ns: null}
calibration: {source: factory, version: v1}
quality: {status: unvalidated}
"""
    )
    return cfg


def test_parser_roundtrip_and_units(tmp_path):
    raw = tmp_path / "legacy.yaml"
    raw.write_text(
        """
cameras:
  - id: CAM_A
    resolution: {width: 800, height: 600}
    model: kb
    intrinsics: {fx: 300, fy: 300, cx: 400, cy: 300}
    extrinsics:
      reference_camera: CAM_B
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation_mm: [120.86, 0, 0]
"""
    )
    calib = load_device_calibration(raw)
    assert calib.cameras[0].model.type == "kannala_brandt_4"
    assert np.allclose(calib.cameras[0].extrinsics.translation_m, [0.12086, 0, 0])


def test_calibration_id_format(config_path):
    calib = load_device_calibration(config_path)
    cid = calib.calibration_id("IR_L")
    assert cid == "TESTSERIAL1:IR_L:1600x1300:v1"


def test_validate_good_calibration(config_path):
    calib = load_device_calibration(config_path)
    results = validate_calibration(calib, roundtrip_samples=20_000)
    for cid, qa in results.items():
        assert qa.status in ("PASS", "DEGRADED"), qa.report.to_json()
        names = {c.name for c in qa.report.checks if c.evaluated}
        assert {"schema", "projection_roundtrip"} <= names


def test_validate_rejects_bad_rotation(tmp_path):
    bad = tmp_path / "badrot.yaml"
    text = """
cameras:
  - id: IR_R
    resolution: {width: 100, height: 100}
    model: {type: pinhole}
    intrinsics: {fx: 80, fy: 80, cx: 50, cy: 50}
    extrinsics:
      reference_camera: IR_L
      rotation: [[1.001, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation_m: [0.12, 0, 0]
"""
    bad.write_text(text)
    calib = load_device_calibration(bad)
    results = validate_calibration(calib, roundtrip_samples=2000)
    check = [c for c in results["IR_R"].report.checks if c.name == "rotation"][0]
    assert not check.passed


def test_baseline_out_of_range_fails(config_path):
    calib = load_device_calibration(config_path)
    calib = calib.model_copy(deep=True)
    calib.cameras[1].extrinsics.translation_m = [0.30, 0, 0]
    results = validate_calibration(calib, baseline_range=(0.10, 0.14), roundtrip_samples=5000)
    baseline_check = [
        c for c in results["IR_R"].report.checks if c.name == "baseline"
    ][0]
    assert not baseline_check.passed


def test_schema_check_flags_negative_fx():
    # schema stays permissive on purpose: load -> QA -> FAIL/QUARANTINE
    from camera_geometry.qa.checks import check_schema

    calib = DeviceCalibration.model_validate(
        {
            "cameras": [
                {
                    "id": "C",
                    "resolution": {"width": 10, "height": 10},
                    "model": {"type": "pinhole"},
                    "intrinsics": {"fx": -1, "fy": 1, "cx": 5, "cy": 5},
                }
            ]
        }
    )
    result = check_schema(calib, calib.camera("C"))
    assert not result.passed
    assert result.metrics["fx_positive"] is False


# ---------------------------------------------------------------- resolver
def test_resolver_priority_factory_vs_registry(config_path, tmp_path):

    calib = load_device_calibration(config_path)
    store = CalibrationStore(tmp_path / "store")
    rec = CalibrationRecord(
        calibration_id=calib.calibration_id("IR_L"),
        calibration=calib,
        quality_status="PASS",
    )
    store.add(rec)
    resolver = CalibrationResolver(calib, store)
    resolved = resolver.resolve("IR_L", (1600, 1300))
    # validated registry entry outranks factory
    assert resolved.source == "validated_device"

    resolver2 = CalibrationResolver(calib, None)
    assert resolver2.resolve("IR_L", (1600, 1300)).source == "factory"

    resolver3 = CalibrationResolver(None, None)
    with pytest.raises(CalibrationNotFound):
        resolver3.resolve("NOPE")


# -------------------------------------------------------------- lut cache
def test_lut_cache_roundtrip(tmp_path):
    cache = LUTCache(tmp_path)
    key = make_key({"a": 1, "b": [1, 2]})
    assert key == make_key({"b": [1, 2], "a": 1})
    assert not cache.has(key, "x")
    arr = np.arange(12).reshape(3, 4).astype(np.float32)
    cache.put(key, "x", {"grid": arr}, meta={"v": 1})
    got = cache.get(key, "x")
    assert got is not None and np.allclose(got["grid"], arr)


# ------------------------------------------------------------ torch passthrough
def test_torch_passthrough_optional():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from camera_geometry.models.kb4 import KannalaBrandt4

    cam = KannalaBrandt4(504.97, 504.85, 825.88, 644.70,
                         0.07305208, -0.01165996, 0.00033907, 0.00027176, 160, 130)
    uv = torch.tensor([[800.0, 650.0]], dtype=torch.float64)
    rays = cam.pixel_to_ray(uv)
    assert isinstance(rays, torch.Tensor)
    back = cam.ray_to_pixel(rays)
    assert isinstance(back, torch.Tensor)
    assert np.allclose(back.numpy(), uv.numpy(), atol=1e-6)

"""End-to-end integration test: raw synthetic stereo fisheye -> full DAG."""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from camera_geometry.calibration.parser import dump_device_calibration
from camera_geometry.pipeline import AutoGeometryPipeline
from camera_geometry.schema import (
    CalibrationMeta,
    CameraEntry,
    DeviceCalibration,
    DeviceInfo,
    Extrinsics,
    Intrinsics,
    KB4Distortion,
    ModelSpec,
    QualityInfo,
    Resolution,
)
from camera_geometry.stereo.depth import stereo_depth
from camera_geometry.testing.synthetic import (
    make_stereo_rig,
    render_textured_room,
    render_textured_room_ss,
)
from camera_geometry.transforms import SE3


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    root = tmp_path_factory.mktemp("episode")
    cam_l, cam_r, T_left_right = make_stereo_rig(width=1600, height=1300, seed=21)
    T_right_left = T_left_right.inverse()

    ep = root / "episode_000001"
    (ep / "raw" / "IR_L").mkdir(parents=True)
    (ep / "raw" / "IR_R").mkdir(parents=True)
    n_frames = 8
    cache_l: dict = {}
    cache_r: dict = {}
    identity_R = np.eye(3)
    for i in range(n_frames):
        shift = np.array([0.01 * i, 0.004 * i, 0.02 * i])
        # camera moved by +shift in room frame => room->cam translation = -shift
        pose_l = SE3.from_Rt(identity_R, -shift)
        pose_r = T_right_left.compose(SE3.from_Rt(identity_R, -shift))
        cv2.imwrite(
            str(ep / "raw" / "IR_L" / f"frame_{i:06d}.png"),
            render_textured_room(cam_l, pose_l, ray_cache=cache_l),
        )
        cv2.imwrite(
            str(ep / "raw" / "IR_R" / f"frame_{i:06d}.png"),
            render_textured_room(cam_r, pose_r, ray_cache=cache_r),
        )

    def entry(cid, cam, ext):
        return CameraEntry(
            id=cid,
            resolution=Resolution(width=cam.width, height=cam.height),
            model=ModelSpec(type="kannala_brandt_4", convention="synthetic_kb_v1"),
            intrinsics=Intrinsics(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy),
            distortion=KB4Distortion(k1=cam.k[0], k2=cam.k[1], k3=cam.k[2], k4=cam.k[3]),
            extrinsics=ext,
        )

    calib = DeviceCalibration(
        device=DeviceInfo(vendor="synthetic", serial_number="ITEST000001"),
        cameras=[
            entry("IR_L", cam_l, None),
            entry("IR_R", cam_r, Extrinsics(
                reference_camera="IR_L",
                rotation=T_left_right.R.tolist(),
                translation_m=T_left_right.t.tolist(),
            )),
        ],
        calibration=CalibrationMeta(source="factory", version="v1"),
        quality=QualityInfo(status="unvalidated"),
    )
    cfg_path = root / "device.yaml"
    dump_device_calibration(calib, cfg_path)
    return {"root": root, "episode": ep, "config": cfg_path}


def test_full_pipeline_passes_acceptance(episode, tmp_path):
    pipe = AutoGeometryPipeline(config_path=episode["config"], home=tmp_path / "geomhome",
                                roundtrip_samples=20_000)
    report = pipe.process_episode(
        input_dir=episode["episode"],
        output_dir=episode["episode"],
        qa_pairs=4,
        write_virtual=True,
        write_rectified=True,
    )
    m = report.metrics
    assert report.status == "PASS", json.dumps(report.to_json(), indent=2)[:2500]
    assert m["projection_roundtrip_p99"] < 0.05
    assert m["stereo_epipolar_median"] < 0.5
    assert m["stereo_epipolar_p95"] < 1.0
    assert m["calibration_health_score"] >= 0.90
    assert m["virtual_camera_coverage"] > 0.9

    out_root = episode["episode"]
    assert (out_root / "geometry" / "calibration_refs.json").exists()
    refs = json.loads((out_root / "geometry" / "calibration_refs.json").read_text())
    assert refs["IR_L"].startswith("ITEST000001:IR_L")
    vdir = out_root / "derived" / "virtual_ego_100" / "IR_L"
    assert vdir.exists() and len(list(vdir.glob("*.png"))) == 8
    assert len(list((out_root / "derived" / "stereo_rectified_IR_L-IR_R" / "left").glob("*.png"))) == 8

    # second run must hit LUT cache and stay PASS
    report2 = pipe.process_episode(input_dir=episode["episode"], output_dir=out_root, qa_pairs=2)
    assert report2.status == "PASS"


def test_quarantine_on_broken_calibration(episode, tmp_path):
    import yaml

    data = yaml.safe_load(episode["config"].read_text())
    data["device"]["serial_number"] = "BROKEN00001"
    data["cameras"][1]["extrinsics"]["translation_m"] = [0.40, 0, 0]
    broken_cfg = tmp_path / "broken.yaml"
    broken_cfg.write_text(yaml.safe_dump(data))

    pipe = AutoGeometryPipeline(config_path=broken_cfg, home=tmp_path / "home2", roundtrip_samples=5000)
    report = pipe.process_episode(input_dir=episode["episode"], output_dir=tmp_path / "out2", qa_pairs=1)
    assert report.status == "QUARANTINE"
    q = report.quarantine
    assert q["reason"] == "camera_geometry_unstable"
    assert q["failure_stage"] == "static_qa"


def test_enable_fallback_override_does_not_mutate_instance(episode, tmp_path):
    """Regression: a per-call enable_fallback=False must not permanently
    disable fallback for subsequent default calls on the same pipeline."""
    pipe = AutoGeometryPipeline(config_path=episode["config"], home=tmp_path / "home_ef",
                                roundtrip_samples=2000)
    assert pipe.enable_fallback is True
    pipe.process_episode(input_dir=episode["episode"], output_dir=tmp_path / "out_ef",
                         qa_pairs=1, enable_fallback=False)
    assert pipe.enable_fallback is True  # instance policy unchanged


def test_multi_camera_episode(tmp_path):
    """3-camera rig: reference + two others -> per-camera views + rectified pair."""
    cam_l, cam_r, T_lr = make_stereo_rig(width=1600, height=1300, seed=51)
    from camera_geometry.testing.synthetic import make_kb4_camera

    cam_l2 = make_kb4_camera(52, 1600, 1300)
    T_l_l2 = SE3.from_Rt(np.eye(3), np.array([-0.06, 0.0, 0.01]))  # IR_L2 center in IR_L frame
    T_l2_l = T_l_l2.inverse()

    ep = tmp_path / "ep3"
    (ep / "raw" / "IR_L").mkdir(parents=True)
    (ep / "raw" / "IR_R").mkdir(parents=True)
    (ep / "raw" / "IR_L2").mkdir(parents=True)
    cache_l, cache_r, cache_l2 = {}, {}, {}
    idn = np.eye(3)
    for i in range(3):
        s = np.array([0.02 * i, 0.01 * i, 0.02 * i])
        cv2.imwrite(str(ep / "raw" / "IR_L" / f"f{i}.png"),
                    render_textured_room_ss(cam_l, SE3.from_Rt(idn, -s), ray_cache=cache_l))
        cv2.imwrite(str(ep / "raw" / "IR_R" / f"f{i}.png"),
                    render_textured_room(cam_r, T_lr.inverse().compose(SE3.from_Rt(idn, -s)), ray_cache=cache_r))
        cv2.imwrite(str(ep / "raw" / "IR_L2" / f"f{i}.png"),
                    render_textured_room(cam_l2, T_l2_l.compose(SE3.from_Rt(idn, -s)), ray_cache=cache_l2))

    def entry(cid, cam, ext):
        return CameraEntry(
            id=cid, resolution=Resolution(width=1600, height=1300),
            model=ModelSpec(type="kannala_brandt_4"),
            intrinsics=Intrinsics(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy),
            distortion=KB4Distortion(k1=cam.k[0], k2=cam.k[1], k3=cam.k[2], k4=cam.k[3]),
            extrinsics=ext,
        )

    calib = DeviceCalibration(
        device=DeviceInfo(vendor="synthetic", serial_number="MULTICAM001"),
        cameras=[
            entry("IR_L", cam_l, None),
            entry("IR_R", cam_r, Extrinsics(reference_camera="IR_L",
                                            rotation=T_lr.R.tolist(),
                                            translation_m=T_lr.t.tolist())),
            entry("IR_L2", cam_l2, Extrinsics(reference_camera="IR_L",
                                              rotation=T_l_l2.R.tolist(),
                                              translation_m=T_l_l2.t.tolist())),
        ],
        calibration=CalibrationMeta(source="factory", version="v1"),
        quality=QualityInfo(status="unvalidated"),
    )
    cfg = tmp_path / "device3.yaml"
    dump_device_calibration(calib, cfg)

    pipe = AutoGeometryPipeline(config_path=cfg, home=tmp_path / "home", roundtrip_samples=10_000,
                                baseline_range=(0.04, 0.16), max_fallback_frames=3)
    report = pipe.process_episode(input_dir=ep, output_dir=ep, qa_pairs=2, write_depth=False)
    assert report.status in ("PASS", "DEGRADED"), json.dumps(report.to_json())[:1500]
    virt = ep / "derived" / "virtual_ego_100"
    for cid in ("IR_L", "IR_R", "IR_L2"):
        assert (virt / cid).exists(), f"missing virtual view for {cid}"
    assert (ep / "derived" / "stereo_rectified_IR_L-IR_R" / "left").exists()


def test_depth_smoke(episode):
    from camera_geometry.runtime.opencv_backend import remap_image
    from camera_geometry.models.factory import build_camera
    from camera_geometry.calibration.parser import load_device_calibration
    from camera_geometry.stereo.rectify import build_stereo_rectifier
    from camera_geometry.transforms import SE3

    calib = load_device_calibration(episode["config"])
    el, er = calib.camera("IR_L"), calib.camera("IR_R")
    cam_l, cam_r = build_camera(el), build_camera(er)
    ext = er.extrinsics
    rig = build_stereo_rectifier(cam_l, cam_r, SE3.from_Rt(ext.rotation, ext.translation_m), 960, 720, 100.0)
    img_l = cv2.imread(str(sorted((episode["episode"] / "raw" / "IR_L").glob("*.png"))[0]))
    img_r = cv2.imread(str(sorted((episode["episode"] / "raw" / "IR_R").glob("*.png"))[0]))
    disp, depth = stereo_depth(remap_image(img_l, rig.grid_left), remap_image(img_r, rig.grid_right),
                               rig.virtual_camera.fx, rig.baseline_m)
    valid = np.isfinite(depth) & (depth > 0)
    ratio = float(valid.mean())
    med = float(np.median(depth[valid])) if valid.any() else float("nan")
    assert ratio > 0.05, f"too few valid disparities: {ratio}"
    assert 0.8 < med < 7.0, f"median depth {med} outside scene range"

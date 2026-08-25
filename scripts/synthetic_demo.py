"""Synthetic end-to-end demo without hardware.

Builds a synthetic KB4 stereo rig (1600x1300, B=120.86mm), renders a textured
room into raw fisheye pairs with an EXACT ray-cast renderer, then runs the
full automatic pipeline: resolve -> QA -> LUTs -> frame QA -> outputs.

Run:  python scripts/synthetic_demo.py [--outdir outputs/demo]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

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
from camera_geometry.testing.synthetic import make_stereo_rig, render_textured_room
from camera_geometry.transforms import SE3

N_FRAMES = 12


def build_config(cam_l, cam_r, T_left_right: SE3, path: Path) -> DeviceCalibration:
    def entry(cid: str, cam, ext: Extrinsics | None) -> CameraEntry:
        return CameraEntry(
            id=cid,
            resolution=Resolution(width=cam.width, height=cam.height),
            model=ModelSpec(type="kannala_brandt_4", convention="synthetic_kb_v1"),
            intrinsics=Intrinsics(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy),
            distortion=KB4Distortion(k1=cam.k[0], k2=cam.k[1], k3=cam.k[2], k4=cam.k[3]),
            extrinsics=ext,
        )

    calib = DeviceCalibration(
        device=DeviceInfo(vendor="synthetic", serial_number="SYNTH000000"),
        cameras=[
            entry("IR_L", cam_l, None),
            entry(
                "IR_R",
                cam_r,
                Extrinsics(
                    reference_camera="IR_L",
                    rotation=T_left_right.R.tolist(),
                    translation_m=T_left_right.t.tolist(),
                ),
            ),
        ],
        calibration=CalibrationMeta(source="factory", version="v1"),
        quality=QualityInfo(status="unvalidated"),
    )
    dump_device_calibration(calib, path)
    return calib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/demo")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    episode = outdir / "episode_000001"
    (episode / "raw" / "IR_L").mkdir(parents=True)
    (episode / "raw" / "IR_R").mkdir(parents=True)

    print("[demo] building synthetic KB4 rig (1600x1300, B=120.86 mm) ...")
    cam_l, cam_r, T_left_right = make_stereo_rig(width=1600, height=1300, seed=11)
    config_path = episode / "device.yaml"
    build_config(cam_l, cam_r, T_left_right, config_path)

    print(f"[demo] rendering {N_FRAMES} textured fisheye pairs (exact ray-cast) ...")
    cache_l: dict = {}
    cache_r: dict = {}
    identity_R = np.eye(3)
    t0 = time.time()
    for i in range(N_FRAMES):
        shift = np.array([0.01 * i, 0.005 * i, 0.03 * i])
        pose_l = SE3.from_Rt(identity_R, -shift)
        pose_r = T_left_right.inverse().compose(SE3.from_Rt(identity_R, -shift))
        cv2.imwrite(str(episode / "raw" / "IR_L" / f"frame_{i:06d}.png"),
                    render_textured_room(cam_l, pose_l, ray_cache=cache_l))
        cv2.imwrite(str(episode / "raw" / "IR_R" / f"frame_{i:06d}.png"),
                    render_textured_room(cam_r, pose_r, ray_cache=cache_r))
    print(f"[demo] rendered in {time.time() - t0:.1f}s")

    print("[demo] running automatic pipeline (no human input) ...")
    pipeline = AutoGeometryPipeline(config_path=config_path, home=outdir / "registry_home",
                                   roundtrip_samples=100_000)
    report = pipeline.process_episode(
        input_dir=episode,
        output_dir=episode,
        qa_pairs=8,
        write_virtual=True,
        write_rectified=True,
    )
    m = report.metrics
    print(json.dumps({
        "status": report.status,
        "calibration_id": m.get("calibration_id"),
        "projection_roundtrip_p99_px": m.get("projection_roundtrip_p99"),
        "stereo_epipolar_median_px": m.get("stereo_epipolar_median"),
        "stereo_epipolar_p95_px": m.get("stereo_epipolar_p95"),
        "virtual_camera_coverage": m.get("virtual_camera_coverage"),
        "calibration_health_score": m.get("calibration_health_score"),
        "runtime_ms": m.get("runtime_ms"),
    }, indent=2))

    checks = {
        "status==PASS": report.status == "PASS",
        "roundtrip P99 < 0.05 px": m.get("projection_roundtrip_p99", 1e9) < 0.05,
        "epipolar median < 0.5 px": m.get("stereo_epipolar_median", 1e9) < 0.5,
        "epipolar p95 < 1.0 px": m.get("stereo_epipolar_p95", 1e9) < 1.0,
        "health score >= 0.90": m.get("calibration_health_score", 0) >= 0.90,
    }
    print("\n[demo] MVP acceptance (spec section 49):")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\noutputs at: {episode}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Performance benchmarks: geometry core, remap, full pipeline.

Run:  python scripts/benchmark.py [--json]
Prints a table; use --json for machine-readable output (CI regression base).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def bench_unproject(n=2_000_000, repeats=3):
    from camera_geometry.models.kb4 import KannalaBrandt4

    cam = KannalaBrandt4(505, 505, 800, 650, 0.072, -0.0117, 0.0003, 0.0002, 1600, 1300)
    rng = np.random.default_rng(0)
    uv = np.column_stack([rng.uniform(0, 1600, n), rng.uniform(0, 1300, n)])
    cam.unproject(uv[:10000])  # warm LUT
    best = min(_time(lambda: cam.unproject(uv)) for _ in range(repeats))
    return {"unproject_2M_s": round(best, 3), "M_pts_per_s": round(n / best / 1e6, 1)}


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench_raymap(repeats=2):
    from camera_geometry.geometry.ray_map import generate_ray_map
    from camera_geometry.models.kb4 import KannalaBrandt4

    cam = KannalaBrandt4(505, 505, 800, 650, 0.072, -0.0117, 0.0003, 0.0002, 1600, 1300)
    return {"raymap_1600x1300_s": round(min(_time(lambda: generate_ray_map(cam)) for _ in range(repeats)), 2)}


def bench_remap(n_frames=100, repeats=2):

    from camera_geometry.models.kb4 import KannalaBrandt4
    from camera_geometry.models.pinhole import PinholeCamera
    from camera_geometry.runtime.opencv_backend import remap_image
    from camera_geometry.virtual_camera.remap import create_remap_grid

    cam = KannalaBrandt4(505, 505, 800, 650, 0.072, -0.0117, 0.0003, 0.0002, 1600, 1300)
    vcam = PinholeCamera.from_fov(1024, 1024, 100.0)
    grid, _ = create_remap_grid(cam, vcam)
    frame = np.random.default_rng(0).integers(0, 255, (1300, 1600, 3), dtype=np.uint8)

    def run():
        for _ in range(n_frames):
            remap_image(frame, grid)

    best = min(_time(run) for _ in range(repeats))
    return {f"remap_{n_frames}frames_s": round(best, 2), "remap_fps": round(n_frames / best, 1)}


def bench_pipeline_episode(frames=12, repeats=1):
    from camera_geometry.pipeline import AutoGeometryPipeline
    from camera_geometry.transforms import SE3
    from camera_geometry.testing.synthetic import make_stereo_rig, render_textured_room
    from camera_geometry.calibration.parser import dump_device_calibration
    from camera_geometry.schema import (
        CalibrationMeta, CameraEntry, DeviceCalibration, DeviceInfo,
        Extrinsics, Intrinsics, KB4Distortion, ModelSpec, QualityInfo, Resolution,
    )
    import cv2

    tmp = Path(tempfile.mkdtemp())
    cam_l, cam_r, T = make_stereo_rig(width=1600, height=1300, seed=77)
    ep = tmp / "ep"
    (ep / "raw" / "IR_L").mkdir(parents=True)
    (ep / "raw" / "IR_R").mkdir(parents=True)
    cache_l, cache_r = {}, {}
    idn = np.eye(3)
    for i in range(frames):
        s = np.array([0.02 * i, 0.01 * i, 0.02 * i])
        cv2.imwrite(str(ep / "raw" / "IR_L" / f"f{i}.png"),
                    render_textured_room(cam_l, SE3.from_Rt(idn, -s), ray_cache=cache_l))
        cv2.imwrite(str(ep / "raw" / "IR_R" / f"f{i}.png"),
                    render_textured_room(cam_r, T.inverse().compose(SE3.from_Rt(idn, -s)),
                                         ray_cache=cache_r))

    def entry(cid, cam, ext):
        return CameraEntry(id=cid, resolution=Resolution(width=1600, height=1300),
                           model=ModelSpec(type="kannala_brandt_4"),
                           intrinsics=Intrinsics(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy),
                           distortion=KB4Distortion(k1=cam.k[0], k2=cam.k[1], k3=cam.k[2], k4=cam.k[3]),
                           extrinsics=ext)

    calib = DeviceCalibration(
        device=DeviceInfo(serial_number="BENCH001"),
        cameras=[entry("IR_L", cam_l, None),
                 entry("IR_R", cam_r, Extrinsics(reference_camera="IR_L",
                                                 rotation=T.R.tolist(), translation_m=T.t.tolist()))],
        calibration=CalibrationMeta(source="factory"), quality=QualityInfo(status="unvalidated"))
    cfg = tmp / "device.yaml"
    dump_device_calibration(calib, cfg)

    times = []
    for _ in range(repeats + 1):  # first pass cold, later warm
        pipe = AutoGeometryPipeline(device_calibration=calib, home=tmp / "home",
                                    roundtrip_samples=20000)
        t0 = time.perf_counter()
        rep = pipe.process_episode(input_dir=ep, output_dir=ep, qa_pairs=2,
                                   write_virtual=True, write_rectified=True)
        times.append(time.perf_counter() - t0)
        assert rep.status in ("PASS", "DEGRADED")
    return {f"pipeline_{frames}frames_cold_s": round(times[0], 1),
            f"pipeline_{frames}frames_warm_s": round(times[-1], 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    args = ap.parse_args()
    results = {}
    results.update(bench_unproject())
    results.update(bench_raymap())
    results.update(bench_remap())
    if not args.skip_pipeline:
        results.update(bench_pipeline_episode())
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        w = max(len(k) for k in results)
        for k, v in results.items():
            print(f"{k:<{w}}  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

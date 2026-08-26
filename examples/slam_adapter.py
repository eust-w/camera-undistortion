"""SLAM adapter example (spec section 24).

SLAM should NOT consume undistorted images. The contract is:

    raw feature pixel  ->  camera.unproject()  ->  bearing vector  ->  SLAM

This example exercises the exact flow on synthetic data:
  1. load factory calibration, build the KB4 model
  2. "detect" features in the raw fisheye frame (here: known 3D points
     projected — in production these come from your feature frontend)
  3. convert to unit bearing vectors and feed a two-view triangulation,
     the same geometry a SLAM backend performs for triangulation/BA
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_geometry.calibration.fitting import fit_model  # noqa: F401  (SLAM BA hooks)
from camera_geometry.geometry.triangulation import triangulate_midpoint
from camera_geometry.models.factory import build_camera
from camera_geometry.schema import (
    CalibrationMeta,
    CameraEntry,
    DeviceCalibration,
    DeviceInfo,
    Extrinsics,
    Intrinsics,
    KB4Distortion,
    ModelSpec,
    Resolution,
)
from camera_geometry.testing.synthetic import make_stereo_rig
from camera_geometry.transforms import SE3


def main() -> int:
    # -- 1. factory calibration -> CameraModel (the only SLAM-facing object)
    cam_l, cam_r, T_left_right = make_stereo_rig(width=1600, height=1300, seed=7)
    calib = DeviceCalibration(
        device=DeviceInfo(serial_number="SLAMDEMO001"),
        cameras=[
            CameraEntry(id="IR_L", resolution=Resolution(width=1600, height=1300),
                        model=ModelSpec(type="kannala_brandt_4"),
                        intrinsics=Intrinsics(fx=cam_l.fx, fy=cam_l.fy, cx=cam_l.cx, cy=cam_l.cy),
                        distortion=KB4Distortion(k1=float(cam_l.k[0]), k2=float(cam_l.k[1]), k3=float(cam_l.k[2]), k4=float(cam_l.k[3]))),
            CameraEntry(id="IR_R", resolution=Resolution(width=1600, height=1300),
                        model=ModelSpec(type="kannala_brandt_4"),
                        intrinsics=Intrinsics(fx=cam_r.fx, fy=cam_r.fy, cx=cam_r.cx, cy=cam_r.cy),
                        distortion=KB4Distortion(k1=float(cam_r.k[0]), k2=float(cam_r.k[1]), k3=float(cam_r.k[2]), k4=float(cam_r.k[3])),
                        extrinsics=Extrinsics(reference_camera="IR_L",
                                              rotation=T_left_right.R.tolist(),
                                              translation_m=T_left_right.t.tolist())),
        ],
        calibration=CalibrationMeta(source="factory"),
    )
    camera = build_camera(calib.camera("IR_L"))

    # -- 2. "feature detections" in the raw fisheye images (pixels)
    rng = np.random.default_rng(0)
    landmarks = rng.uniform([-2, -1, 2.0], [2, 1, 6.0], (400, 3))  # world == left frame
    uv_l = np.asarray(camera.project(landmarks))
    uv_r = np.asarray(cam_r.project(T_left_right.inverse().apply(landmarks)))

    # -- 3. pixels -> bearing vectors (THE SLAM adapter call)
    bearings_l = np.asarray(camera.pixel_to_ray(uv_l))
    bearings_r = np.asarray(cam_r.pixel_to_ray(uv_r))
    assert np.allclose(np.linalg.norm(bearings_l, axis=-1), 1.0)

    # -- 4. two-view triangulation from bearings (what the SLAM backend does)
    points, residuals = triangulate_midpoint(bearings_l, bearings_r, T_left_right)
    err = np.linalg.norm(points - landmarks, axis=-1)
    good = np.isfinite(residuals) & (residuals < 0.01)
    print(f"[slam] triangulated {good.sum()}/{len(landmarks)} landmarks")
    print(f"[slam] position error  median={np.median(err[good])*100:.3f} cm  p95={np.percentile(err[good],95)*100:.3f} cm")
    ok = good.sum() > 300 and np.median(err[good]) < 1e-4
    print("[slam] OK" if ok else "[slam] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

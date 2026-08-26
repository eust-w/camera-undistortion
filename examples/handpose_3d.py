"""Hand / Object 3D adapter example (spec section 27).

Flow:
    raw fisheye  ->  virtual pinhole view  ->  2D detector (your model)
               ->  virtual pixel
    virtual pixel ->  virtual ray  ->  camera-frame ray (T between frames)
    left ray + right ray + T_left_right  ->  triangulation  ->  3D keypoints

The 2D "detections" here are produced by projecting synthetic 3D joints into
both virtual views and adding pixel noise — swap in your detector output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_geometry.models.factory import build_camera  # noqa: E402
from camera_geometry.schema import (  # noqa: E402
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
from camera_geometry.geometry.triangulation import triangulate_midpoint  # noqa: E402
from camera_geometry.testing.synthetic import make_stereo_rig  # noqa: E402
from camera_geometry.transforms import SE3  # noqa: E402
from camera_geometry.virtual_camera.presets import create_virtual_camera  # noqa: E402
from camera_geometry.virtual_camera.remap import create_remap_grid  # noqa: E402


def main() -> int:
    cam_l, cam_r, T_left_right = make_stereo_rig(width=1600, height=1300, seed=13)
    calib = DeviceCalibration(
        device=DeviceInfo(serial_number="HANDDEMO001"),
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
    raw_l = build_camera(calib.camera("IR_L"))
    raw_r = build_camera(calib.camera("IR_R"))
    vcam = create_virtual_camera("ego_100")  # the 2D detector's input space
    grid_l, _ = create_remap_grid(raw_l, vcam)
    grid_r, _ = create_remap_grid(raw_r, vcam)

    # -- synthetic 3D "hand" joints (21 keypoints in front of the rig)
    t = np.linspace(0, 1, 21)
    joints3d = np.column_stack([
        0.3 * np.sin(3 * t), 0.2 * np.cos(4 * t) - 0.2, 1.2 + 0.8 * t
    ])

    # -- "detector": joints appear in the VIRTUAL views (ego_100); the virtual
    # view is a pure reparametrization of the raw rays, so the virtual pixel
    # of a 3D point is vcam.project(ray through the raw camera).
    def detect(cam_raw, joints, seed):
        rays = np.asarray(cam_raw.unproject(
            np.asarray(cam_raw.project(joints))))  # raw pixels -> raw rays
        virt_px = np.asarray(vcam.project(rays))   # detector input space
        r = np.random.default_rng(seed)
        return virt_px + r.normal(0, 1.0, virt_px.shape)  # ~1 px detector noise

    det_l = detect(raw_l, joints3d, 0)
    det_r = detect(raw_r, T_left_right.inverse().apply(joints3d), 1)  # joints -> right frame

    # -- THE adapter math: virtual pixel -> virtual ray -> camera-frame ray
    rays_l = np.asarray(vcam.pixel_to_ray(det_l))
    rays_r = np.asarray(vcam.pixel_to_ray(det_r))

    # -- triangulate bearings in the raw left camera frame
    points, residuals = triangulate_midpoint(rays_l, rays_r, T_left_right)

    # report in metric units
    err = np.linalg.norm(points - joints3d, axis=-1)
    ok_pts = np.isfinite(err) & (err < 0.5)
    depth = joints3d[:, 2]
    print(f"[hand3d] {ok_pts.sum()}/{len(joints3d)} joints triangulated")
    print(f"[hand3d] 3D error  median={np.median(err[ok_pts])*100:.2f} cm "
          f"@ depth {depth.min():.1f}-{depth.max():.1f} m (1 px detector noise)")
    # theoretical stereo depth error at 1 px noise: z^2*sigma/(f*B)
    # ~ 3.7 cm @ 1.2 m .. 10 cm @ 2.0 m for this rig
    ok = ok_pts.sum() >= 18 and np.median(err[ok_pts]) < 0.06
    print("[hand3d] OK" if ok else "[hand3d] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

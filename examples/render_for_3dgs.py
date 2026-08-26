"""3DGS / NeRF adapter example (spec section 25).

Best path: the renderer supports generic cameras —

    Gaussian  ->  camera.project()  ->  raw fisheye pixel

or ray-based rendering through the precomputed ray map:

    pixel  ->  ray_map  ->  Gaussian evaluation along the ray

This example renders a synthetic point cloud into the RAW fisheye image via
`camera.project()` and provides the ray map + valid mask a ray-based renderer
consumes — full FOV preserved, no undistortion anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from camera_geometry.geometry.ray_map import generate_ray_map, sample_valid_pixels  # noqa: E402
from camera_geometry.models.factory import build_camera  # noqa: E402
from camera_geometry.schema import (  # noqa: E402
    CameraEntry,
    DeviceCalibration,
    DeviceInfo,
    Intrinsics,
    KB4Distortion,
    ModelSpec,
    Resolution,
)
from camera_geometry.testing.synthetic import make_stereo_rig  # noqa: E402


def main() -> int:
    cam_l, _, _ = make_stereo_rig(width=1600, height=1300, seed=9)
    calib = DeviceCalibration(
        device=DeviceInfo(serial_number="GSDEMO001"),
        cameras=[CameraEntry(
            id="IR_L", resolution=Resolution(width=1600, height=1300),
            model=ModelSpec(type="kannala_brandt_4"),
            intrinsics=Intrinsics(fx=cam_l.fx, fy=cam_l.fy, cx=cam_l.cx, cy=cam_l.cy),
            distortion=KB4Distortion(k1=float(cam_l.k[0]), k2=float(cam_l.k[1]), k3=float(cam_l.k[2]), k4=float(cam_l.k[3])),
        )],
    )
    camera = build_camera(calib.camera("IR_L"))

    # -- path A (projection): "Gaussians" -> raw fisheye pixels
    rng = np.random.default_rng(1)
    centers = rng.uniform([-2.5, -1.5, 1.5], [2.5, 1.5, 5.0], (30000, 3))
    colors = rng.integers(60, 255, (30000, 3), dtype=np.uint8)
    uv = np.asarray(camera.project(centers.astype(np.float64)))
    inside = camera.contains(uv)

    img = np.zeros((cam_l.height, cam_l.width, 3), dtype=np.uint8)
    order = np.argsort(-centers[inside, 2])
    uvi = uv[inside][order].astype(int)
    cols = colors[inside][order]
    for (u, v), c in zip(uvi, cols):
        img[v, u] = c
    cv2.imwrite("outputs/gs_projection_raw.png", img)

    # -- path B (ray-based): ray map + valid mask for ray rendering
    rays, valid = generate_ray_map(camera)
    sample_px = sample_valid_pixels(valid, 8)
    sample_rays = np.asarray(camera.pixel_to_ray(sample_px))

    print(f"[3dgs] path A: {int(inside.sum())} gaussians projected into RAW fisheye "
          f"-> outputs/gs_projection_raw.png")
    print(f"[3dgs] path B: ray map {rays.shape} fp32, valid ratio {valid.mean():.3f}")
    print(f"[3dgs] sample ray-map rays (unit): {np.allclose(np.linalg.norm(sample_rays, axis=-1), 1)}")
    ok = inside.sum() > 20000 and valid.mean() > 0.9
    print("[3dgs] OK" if ok else "[3dgs] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

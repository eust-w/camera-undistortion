"""Golden reference export (spec section 44).

Generates reference pixel->ray / ray->pixel data for regression testing.

Sources:
  * synthetic (default): self-consistency baseline from the current KB4
    implementation -- protects against future code changes (regression).
  * orbbec_sdk: vendor truth. Requires the vendor SDK; implement
    `orbbec_pixel_to_ray(camera_id, uv)` below for your SDK version and run
    with --source orbbec_sdk. This is the Phase-0 convention validation.

Usage:
  python scripts/export_golden.py --source synthetic --tag synth_v1
  python scripts/export_golden.py --source orbbec_sdk --tag orbbec_ch5lb5400j5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_geometry.models.factory import build_camera
from camera_geometry.schema import (
    CameraEntry,
    DeviceCalibration,
    DeviceInfo,
    Intrinsics,
    KB4Distortion,
    ModelSpec,
    Resolution,
)

FACTORY_IR_L = dict(fx=504.97497043, fy=504.85320543, cx=825.88761829, cy=644.70496968,
                    k1=0.07305208, k2=-0.01165996, k3=0.00033907, k4=0.00027176,
                    width=1600, height=1300)


def orbbec_pixel_to_ray(camera_id: str, uv: np.ndarray) -> np.ndarray:
    """IMPLEMENT ME with the vendor SDK (Phase 0).

    Example sketch (pseudo):
        import pyorbbecsdk as ob
        pipe = ob.Pipeline(); profile = ...
        # map each pixel through the device depth->world or intrinsics API
        # and return unit rays in the camera frame.
    """
    raise NotImplementedError("wire up the Orbbec SDK here for vendor-truth golden data")


def export(source: str, tag: str, n_pixels: int, out_root: Path) -> Path:
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)

    entry = CameraEntry(
        id="IR_L",
        resolution=Resolution(width=FACTORY_IR_L["width"], height=FACTORY_IR_L["height"]),
        model=ModelSpec(type="kannala_brandt_4", convention="orbbec_kb_v1"),
        intrinsics=Intrinsics(fx=FACTORY_IR_L["fx"], fy=FACTORY_IR_L["fy"],
                              cx=FACTORY_IR_L["cx"], cy=FACTORY_IR_L["cy"]),
        distortion=KB4Distortion(k1=FACTORY_IR_L["k1"], k2=FACTORY_IR_L["k2"],
                                 k3=FACTORY_IR_L["k3"], k4=FACTORY_IR_L["k4"]),
    )
    calib = DeviceCalibration(device=DeviceInfo(vendor="orbbec", serial_number="CH5LB5400J5"),
                              cameras=[entry])
    import yaml

    (out / "config.yaml").write_text(yaml.safe_dump(calib.model_dump(), sort_keys=False))

    cam = build_camera(entry)
    mask = cam.valid_mask()
    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(20240101)
    idx = rng.choice(len(xs), min(n_pixels, len(xs)), replace=False)
    pixels = np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float64)
    rays = np.asarray(cam.unproject(pixels))
    rays_back = np.asarray(cam.project(rays))

    if source == "orbbec_sdk":
        rays_ref = orbbec_pixel_to_ray("IR_L", pixels)
        rays_ref /= np.linalg.norm(rays_ref, axis=-1, keepdims=True)
        np.savez_compressed(out / "pixel_to_ray.npz", pixels=pixels, rays=rays_ref,
                            source="orbbec_sdk")
    else:
        np.savez_compressed(out / "pixel_to_ray.npz", pixels=pixels, rays=rays,
                            source="synthetic")
    np.savez_compressed(out / "ray_to_pixel.npz", rays=rays, pixels=rays_back,
                        source=source)
    print(f"[golden] wrote {out} ({len(pixels)} pixels, source={source})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "orbbec_sdk"], default="synthetic")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-pixels", type=int, default=20000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tests" / "golden"))
    args = ap.parse_args()
    tag = args.tag or f"{args.source}_default"
    export(args.source, tag, args.n_pixels, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

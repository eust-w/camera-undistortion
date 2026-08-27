"""Import a TUM-VI Euroc-format subset into our episode layout + canonical YAML.

Source: basalt's official DoubleSphere calibration for TUM-VI 512
(data/tumvi_512_ds_calib.json) + extracted mav0/cam{0,1}/data frames.

Produces:
  <episode>/raw/IR_L/*.png   (cam0, paired by exact timestamp)
  <episode>/raw/IR_R/*.png   (cam1)
  <episode>/device.yaml      (canonical schema, DoubleSphere model)

Usage:
  python scripts/import_tumvi.py --root third_party/tumvi/room1 \
      --calib third_party/tumvi/tumvi_512_ds_calib.json \
      [--out outputs/tumvi_episode]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_geometry.calibration.parser import dump_device_calibration  # noqa: E402
from camera_geometry.schema import (  # noqa: E402
    CalibrationMeta,
    CameraEntry,
    DeviceCalibration,
    DeviceInfo,
    Extrinsics,
    Intrinsics,
    ModelSpec,
    QualityInfo,
    Resolution,
    ShutterInfo,
)


def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted subset containing mav0/")
    ap.add_argument("--calib", required=True, help="basalt calib json")
    ap.add_argument("--out", default="outputs/tumvi_episode")
    ap.add_argument("--max-pairs", type=int, default=12)
    args = ap.parse_args()

    root = Path(args.root)
    calib_json = json.loads(Path(args.calib).read_text())["value0"]
    T_imu_cam = [
        SE3_from_quat(c["px"], c["py"], c["pz"], c["qx"], c["qy"], c["qz"], c["qw"])
        for c in calib_json["T_imu_cam"]
    ]
    intr = calib_json["intrinsics"]
    resolution = calib_json['resolution'][0]  # [[w,h],[w,h]] per camera

    # T_cam0_cam1 = inv(T_ic0) @ T_ic1  (both are cam->IMU/body)
    T_c0_c1 = T_imu_cam[0].inverse().compose(T_imu_cam[1])
    baseline = float(np.linalg.norm(T_c0_c1.t))
    print(f"[tumvi] stereo baseline: {baseline*100:.2f} cm")

    def ds_entry(cid: str, idx: int, ext) -> CameraEntry:
        p = intr[idx]["intrinsics"]
        return CameraEntry(
            id=cid,
            resolution=Resolution(width=int(resolution[0]), height=int(resolution[1])),
            model=ModelSpec(type="double_sphere",
                            convention="basalt_tumvi_v1",
                            params={"xi": float(p["xi"]), "alpha": float(p["alpha"])}),
            intrinsics=Intrinsics(fx=float(p["fx"]), fy=float(p["fy"]),
                                  cx=float(p["cx"]), cy=float(p["cy"])),
            extrinsics=ext,
        )

    calib_out = DeviceCalibration(
        device=DeviceInfo(vendor="intel/tum-vi", serial_number="TUMVI512ROOM1"),
        cameras=[
            ds_entry("IR_L", 0, None),
            ds_entry("IR_R", 1, Extrinsics(
                reference_camera="IR_L",
                rotation=T_c0_c1.R.tolist(),
                translation_m=T_c0_c1.t.tolist(),
            )),
        ],
        shutter=ShutterInfo(type="global"),
        calibration=CalibrationMeta(source="factory", version="basalt_tumvi_ds"),
        quality=QualityInfo(status="unvalidated"),
    )

    out = Path(args.out)
    (out / "raw" / "IR_L").mkdir(parents=True, exist_ok=True)
    (out / "raw" / "IR_R").mkdir(parents=True, exist_ok=True)
    dump_device_calibration(calib_out, out / "device.yaml")

    cam0_dir = root / "mav0" / "cam0" / "data"
    cam1_dir = root / "mav0" / "cam1" / "data"
    stamps = sorted({p.stem for p in cam0_dir.glob("*.png")} & {p.stem for p in cam1_dir.glob("*.png")})
    stamps = stamps[: args.max_pairs]
    if len(stamps) == 0:
        print("[tumvi] no timestamp-matched pairs found")
        return 1
    for s in stamps:
        shutil.copy(cam0_dir / f"{s}.png", out / "raw" / "IR_L" / f"{s}.png")
        shutil.copy(cam1_dir / f"{s}.png", out / "raw" / "IR_R" / f"{s}.png")
    print(f"[tumvi] episode at {out}: {len(stamps)} stereo pairs")
    print(f"[tumvi] model: double_sphere(xi={intr[0]['intrinsics']['xi']:.4f}, "
          f"alpha={intr[0]['intrinsics']['alpha']:.4f}) {resolution[0]}x{resolution[1]}")
    return 0


def SE3_from_quat(px, py, pz, qx, qy, qz, qw):
    from camera_geometry.transforms import SE3

    R = quat_to_R(qx, qy, qz, qw)
    return SE3.from_Rt(R, [px, py, pz])


if __name__ == "__main__":
    sys.exit(main())

"""Command line interface: camera-geometry <command> ..."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .calibration.parser import load_device_calibration
from .calibration.validator import validate_calibration
from .geometry.ray_map import save_ray_map
from .models.factory import build_camera
from .pipeline import AutoGeometryPipeline
from .runtime.opencv_backend import remap_image
from .stereo.rectify import build_stereo_rectifier
from .virtual_camera.presets import create_virtual_camera
from .virtual_camera.remap import create_remap_grid, remap_coverage


def _load(path: str):
    return load_device_calibration(path)


def cmd_inspect_camera(args) -> int:
    calib = _load(args.config)
    for entry in calib.cameras:
        cam = build_camera(entry)
        mask = cam.valid_mask()
        info = {
            "camera": entry.id,
            "model": entry.model.type,
            "convention": entry.model.convention,
            "resolution": [cam.width, cam.height],
            "intrinsics": cam.intrinsics_dict(),
            "distortion": cam.distortion_dict(),
            "valid_pixel_ratio": round(float(mask.mean()), 4),
        }
        if hasattr(cam, "fov_deg"):
            info["fov_deg"] = round(cam.fov_deg(), 2)
        if entry.extrinsics is not None:
            info["extrinsics_reference"] = entry.extrinsics.reference_camera
            info["translation_m"] = entry.extrinsics.translation_m
        print(json.dumps(info, indent=2))
    return 0


def cmd_validate_calibration(args) -> int:
    calib = _load(args.config)
    results = validate_calibration(
        calib, baseline_range=tuple(args.baseline_range), roundtrip_samples=args.samples
    )
    payload = {cid: qa.report.to_json() for cid, qa in results.items()}
    text = json.dumps(payload, indent=2)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text)
    print(text)
    worst = min(qa.score for qa in results.values())
    return 0 if all(qa.status in ("PASS",) or qa.status == "DEGRADED" for qa in results.values()) and worst >= 0.75 else 1


def cmd_generate_raymap(args) -> int:
    calib = _load(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = calib.cameras if args.camera is None else [calib.camera(args.camera)]
    for entry in entries:
        cam = build_camera(entry)
        from .geometry.ray_map import generate_ray_map

        rays, valid = generate_ray_map(cam)
        path = out_dir / f"raymap_{entry.id}.npz"
        save_ray_map(path, rays, valid, fp16=not args.fp32)
        print(json.dumps({
            "camera": entry.id,
            "path": str(path),
            "shape": list(rays.shape),
            "dtype": "float16" if not args.fp32 else "float32",
            "bytes_on_disk": path.stat().st_size,
            "valid_pixel_ratio": round(float(valid.mean()), 4),
        }))
    return 0


def cmd_virtual(args) -> int:
    calib = _load(args.config)
    entry = calib.camera(args.camera)
    cam = build_camera(entry)
    vcam = create_virtual_camera(args.preset)
    grid, valid = create_remap_grid(cam, vcam)
    img = remap_image(_imread_or_fail(args.image), grid)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(out, img)
    print(json.dumps({
        "preset": args.preset,
        "coverage": round(remap_coverage(valid), 4),
        "output": str(out),
        "size": [vcam.width, vcam.height],
    }))
    return 0


def cmd_rectify(args) -> int:
    calib = _load(args.config)
    el = calib.camera(args.camera_l)
    er = calib.camera(args.camera_r)
    cam_l, cam_r = build_camera(el), build_camera(er)
    T = None
    for e in (el, er):
        if e.extrinsics is not None and e.id != e.extrinsics.reference_camera:
            T = e.extrinsics
    if T is None:
        raise SystemExit("no extrinsics found in config")
    from .transforms import SE3

    rig = build_stereo_rectifier(cam_l, cam_r, SE3.from_Rt(T.rotation, T.translation_m),
                                 args.width, args.height, args.fov)
    img_l = _imread_or_fail(args.left)
    img_r = _imread_or_fail(args.right)
    rect_l = remap_image(img_l, rig.grid_left)
    rect_r = remap_image(img_r, rig.grid_right)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _imwrite(out / "rectified_left.png", rect_l)
    _imwrite(out / "rectified_right.png", rect_r)
    result = {
        "baseline_m": rig.baseline_m,
        "fx_rectified": rig.virtual_camera.fx,
        "outputs": [str(out / "rectified_left.png"), str(out / "rectified_right.png")],
    }
    if args.depth:
        from .stereo.depth import stereo_depth

        disp, depth = stereo_depth(rect_l, rect_r, rig.virtual_camera.fx, rig.baseline_m)
        np.savez_compressed(out / "depth.npz", disparity=disp, depth=depth,
                            fx=rig.virtual_camera.fx, baseline_m=rig.baseline_m)
        valid = np.isfinite(depth)
        result["depth_valid_ratio"] = round(float(valid.mean()), 4)
        result["depth_median_m"] = float(np.median(depth[valid])) if valid.any() else None
    print(json.dumps(result, indent=2))
    return 0


def cmd_process_dataset(args) -> int:
    from .logging_utils import setup_logging

    setup_logging(logging.INFO)
    pipeline = AutoGeometryPipeline(config_path=args.config, home=args.home)
    report = pipeline.process_episode(
        input_dir=args.input,
        output_dir=args.output,
        preset=args.preset,
        max_frames=args.frames,
        write_virtual=not args.no_virtual,
        write_rectified=not args.no_stereo,
        write_depth=args.depth,
    )
    print(json.dumps(report.to_json(), indent=2))
    return 0 if report.status in ("PASS", "DEGRADED") else 1


def _imread_or_fail(path: str):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image: {path}")
    return img


def _imwrite(path: Path, img):
    import cv2

    cv2.imwrite(str(path), img)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="camera-geometry", description="Auto Camera Geometry toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect-camera", help="show parsed canonical camera info")
    s.add_argument("--config", required=True)
    s.set_defaults(func=cmd_inspect_camera)

    s = sub.add_parser("validate-calibration", help="run static QA + health score")
    s.add_argument("--config", required=True)
    s.add_argument("--samples", type=int, default=100_000)
    s.add_argument("--baseline-range", nargs=2, type=float, default=[0.10, 0.14], metavar=("LO", "HI"))
    s.add_argument("--json-out", default=None)
    s.set_defaults(func=cmd_validate_calibration)

    s = sub.add_parser("generate-raymap", help="generate cached ray map + valid mask")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--camera", default=None)
    s.add_argument("--fp32", action="store_true")
    s.set_defaults(func=cmd_generate_raymap)

    s = sub.add_parser("virtual", help="remap one raw fisheye image to virtual pinhole view")
    s.add_argument("--config", required=True)
    s.add_argument("--camera", default="IR_L")
    s.add_argument("--image", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--preset", default="ego_100")
    s.set_defaults(func=cmd_virtual)

    s = sub.add_parser("rectify", help="rectify a stereo fisheye pair")
    s.add_argument("--config", required=True)
    s.add_argument("--camera-l", default="IR_L")
    s.add_argument("--camera-r", default="IR_R")
    s.add_argument("--left", required=True)
    s.add_argument("--right", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=640)
    s.add_argument("--fov", type=float, default=100.0)
    s.add_argument("--depth", action="store_true")
    s.set_defaults(func=cmd_rectify)

    s = sub.add_parser("process-dataset", help="full automatic episode DAG")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--home", default=None)
    s.add_argument("--preset", default="ego_100")
    s.add_argument("--frames", type=int, default=None)
    s.add_argument("--no-virtual", action="store_true")
    s.add_argument("--no-stereo", action="store_true")
    s.add_argument("--depth", action="store_true")
    s.set_defaults(func=cmd_process_dataset)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

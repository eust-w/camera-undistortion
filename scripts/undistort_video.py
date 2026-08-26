"""Undistort a fisheye video into a virtual pinhole view.

Uses factory KB4 calibration (Orbbec IR_L) + cached remap LUT.
If the video resolution differs from the calibration, intrinsics are scaled.

Usage:
  python scripts/undistort_video.py INPUT.mp4 [--out OUT.mp4]
      [--camera IR_L] [--preset ego_100] [--size 1024]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_geometry.models.kb4 import KannalaBrandt4
from camera_geometry.models.pinhole import PinholeCamera
from camera_geometry.virtual_camera.remap import create_remap_grid, to_cv_maps

FACTORY = {
    "IR_L": dict(fx=504.97497043, fy=504.85320543, cx=825.88761829, cy=644.70496968,
                 k1=0.07305208, k2=-0.01165996, k3=0.00033907, k4=0.00027176,
                 width=1600, height=1300),
    "IR_R": dict(fx=505.10234512, fy=504.98765432, cx=826.10451234, cy=645.22109876,
                 k1=0.07311234, k2=-0.01160221, k3=0.00035123, k4=0.00028456,
                 width=1600, height=1300),
}


def build_camera(camera_id: str, video_w: int, video_h: int) -> KannalaBrandt4:
    p = FACTORY[camera_id]
    sx, sy = video_w / p["width"], video_h / p["height"]
    return KannalaBrandt4(p["fx"] * sx, p["fy"] * sy, p["cx"] * sx, p["cy"] * sy,
                          p["k1"], p["k2"], p["k3"], p["k4"], video_w, video_h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--camera", default="IR_L", choices=list(FACTORY))
    ap.add_argument("--preset", default=None, help="ego_90 / ego_100 / ego_110")
    ap.add_argument("--size", type=int, default=1024, help="output square size")
    ap.add_argument("--fov", type=float, default=100.0, help="used when --preset not given")
    ap.add_argument("--backend", choices=["auto", "cv2", "torch"], default="auto")
    args = ap.parse_args()

    src = Path(args.input).expanduser()
    out = Path(args.out).expanduser() if args.out else src.with_name(f"{src.stem}_undistorted.mp4")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {src.name}: {W}x{H} @ {fps:.2f}fps, {total} frames")

    cam = build_camera(args.camera, W, H)
    fov = {"ego_90": 90.0, "ego_100": 100.0, "ego_110": 110.0}.get(args.preset, args.fov)
    vcam = PinholeCamera.from_fov(args.size, args.size, fov)
    t0 = time.time()
    grid, valid = create_remap_grid(cam, vcam)
    mapx, mapy = to_cv_maps(grid)
    coverage = float(valid.mean())
    print(f"[lut] virtual {args.size}x{args.size} fov={fov}deg, coverage={coverage:.3f}, "
          f"built in {time.time()-t0:.1f}s")

    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.size, args.size))
    if not writer.isOpened():
        raise SystemExit("cannot open VideoWriter")

    backend = args.backend
    torch_ctx = None
    if backend in ("auto", "torch"):
        try:
            import torch

            from camera_geometry.runtime.torch_backend import make_sampling_grid, remap_batch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if backend == "torch" or device == "cuda":
                sampling = make_sampling_grid(grid, W, H, device=torch.device(device))
                torch_ctx = (torch, sampling, device)
                print(f"[backend] torch grid_sample on {device}")
        except ImportError:
            if backend == "torch":
                raise SystemExit("torch not installed")
    if torch_ctx is None:
        print("[backend] cv2.remap (CPU)")

    sample_raw = sample_out = None
    n = 0
    t0 = time.time()
    if torch_ctx is not None:
        torch, sampling, device = torch_ctx
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tensor = torch.from_numpy(frame.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            tensor = tensor.to(device)
            out_t = remap_batch(tensor, sampling)
            undist = (out_t[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            writer.write(undist)
            if n == max(0, total // 2):
                sample_raw, sample_out = frame.copy(), undist.copy()
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{total} frames ({n/(time.time()-t0):.1f} fps)")
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            undist = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
            writer.write(undist)
            if n == max(0, total // 2):
                sample_raw, sample_out = frame.copy(), undist.copy()
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{total} frames ({n/(time.time()-t0):.1f} fps)")
    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"[done] {n} frames in {elapsed:.1f}s ({n/elapsed:.1f} fps)")

    raw_small = cv2.resize(sample_raw, (int(512 * W / H), 512))
    out_small = cv2.resize(sample_out, (512, 512))
    side = np.hstack([raw_small, out_small])
    sample_path = out.with_name(f"{out.stem}_sample.jpg")
    cv2.imwrite(str(sample_path), side)

    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True).stdout.strip() != ""
    if has_audio:
        print("[mux] copying original audio track ...")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-i", str(src),
                        "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy",
                        str(out)], check=True)
        tmp.unlink()
    else:
        tmp.replace(out)

    size_mb = out.stat().st_size / 1e6
    print(f"[out] {out} ({size_mb:.1f} MB)")
    print(f"[out] sample: {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

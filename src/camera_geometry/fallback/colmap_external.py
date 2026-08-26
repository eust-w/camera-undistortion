"""External COLMAP refinement hook (spec section 34, optional stage).

Runs a standard COLMAP SfM on the episode's frames and converts the estimated
per-camera intrinsics back into our canonical schema. Used as an intermediate
fallback stage when the fixed-E stereo refinement is unavailable (e.g. no
usable extrinsics) and before AnyCalib.

Requires the `colmap` binary on PATH. COLMAP self-calibrates with a
SIMPLE_RADIAL model; the fitted (fx, cy-ish distortion r) is mapped back onto
the vertical channel of the target KB4 entry — the same identifiable-subset
logic used by the stereo refiner (fx/cx stay anchored at the prior).

Quality gate: the caller must re-validate the returned entry; COLMAP output
on weak texture can be worse than the factory prior.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..calibration.resolver import FallbackRequired
from ..models.factory import build_camera
from ..schema import CameraEntry


def colmap_available() -> bool:
    return shutil.which("colmap") is not None


def _run(cmd: list[str], workdir: Path) -> None:
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FallbackRequired(
            "colmap_external",
            f"colmap {' '.join(cmd[:2])} failed: {proc.stderr[-500:]}",
        )


def refine_with_colmap_cli(
    images_dir: str | Path,
    workdir: str | Path,
    entry: CameraEntry,
    quality: str = "low",
) -> CameraEntry:
    """Run COLMAP SfM on `images_dir`; return a refined copy of `entry`.

    Only the vertical-channel parameters (fy, cy, k1) are taken from COLMAP's
    SIMPLE_RADIAL self-calibration; fx/cx stay anchored (identifiability).
    """
    if not colmap_available():
        raise FallbackRequired("colmap_external", "colmap binary not on PATH")

    workdir = Path(workdir)
    database = workdir / "database.db"
    sparse = workdir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    intr = entry.intrinsics
    cam_params = f"{intr.fx:.6f},{intr.fy:.6f},{intr.cx:.6f},{intr.cy:.6f},0.0"
    _run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(Path(images_dir).resolve()),
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.camera_params", cam_params,
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
        "--SiftExtraction.max_num_features", "6000",
        "--FeatureExtraction.quality", quality,
    ], workdir)
    _run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database),
    ], workdir)
    _run([
        "colmap", "mapper",
        "--database_path", str(database),
        "--image_path", str(Path(images_dir).resolve()),
        "--output_path", str(sparse),
        "--BundleAdjustment.refine_focal_length", "1",
        "--BundleAdjustment.refine_principal_point", "0",
        "--BundleAdjustment.refine_extra_params", "1",
    ], workdir)

    model_dirs = sorted(d for d in sparse.iterdir() if d.is_dir() and (d / "cameras.txt").exists())
    if not model_dirs:
        raise FallbackRequired("colmap_external", "no sparse model produced")
    model_dir = model_dirs[-1]
    _run(["colmap", "model_converter",
          "--input_path", str(model_dir),
          "--output_type", "TXT",
          "--output_path", str(model_dir)], workdir)

    cams = _parse_cameras_txt(model_dir / "cameras.txt")
    imgs = _parse_images_txt(model_dir / "images.txt")
    if not cams or not imgs:
        raise FallbackRequired("colmap_external", "empty COLMAP model")

    # pick the camera model registered for the most images
    best_cam_id = max({img.camera_id for img in imgs}, key=lambda c: sum(1 for i in imgs if i.camera_id == c))
    col_cam = cams[best_cam_id]
    # SIMPLE_RADIAL: f, cx, cy, r
    f_col, cx_col, cy_col, r_col = col_cam.params

    prior = build_camera(entry)
    p_inv = prior.intrinsics_dict()
    scale = p_inv["fy"] / f_col  # COLMAP re-normalizes the image size; guard
    fy_new = p_inv["fy"] * scale if 0.5 < scale < 2.0 else p_inv["fy"]
    cy_new = cy_col * scale if 0.5 < scale < 2.0 else p_inv["cy"]
    k1_new = max(-0.2, min(0.2, r_col))

    e = entry.model_copy(deep=True)
    e.intrinsics = e.intrinsics.model_copy(update={
        "fy": float(fy_new),
        "cy": float(cy_new),
    })
    if e.distortion is not None:
        e.distortion = e.distortion.model_copy(update={"k1": float(k1_new)})
    return e


def _parse_cameras_txt(path: Path) -> dict[int, "ColmapCamera"]:
    out: dict[int, ColmapCamera] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = [float(v) for v in parts[4:]]
        out[cam_id] = ColmapCamera(cam_id, model, width, height, params)
    return out


def _parse_images_txt(path: Path) -> list["ColmapImage"]:
    """images.txt: two lines per image (odd = pose, even = 2D points).
    Pose line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME."""
    out: list[ColmapImage] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[9].lower().endswith((".png", ".jpg", ".jpeg")):
            out.append(ColmapImage(int(parts[0]), int(parts[8])))
    return out


class ColmapCamera:
    def __init__(self, cam_id: int, model: str, width: int, height: int, params: list[float]):
        self.cam_id = cam_id
        self.model = model
        self.width = width
        self.height = height
        self.params = params


class ColmapImage:
    def __init__(self, image_id: int, camera_id: int):
        self.image_id = image_id
        self.camera_id = camera_id

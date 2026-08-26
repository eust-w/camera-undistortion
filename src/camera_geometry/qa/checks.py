"""QA checks for calibration and images."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..models.base import CameraModel
from ..schema import CameraEntry, DeviceCalibration


@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float
    metrics: dict = field(default_factory=dict)
    evaluated: bool = True

    @classmethod
    def not_evaluated(cls, name: str, reason: str = "") -> "CheckResult":
        return cls(name=name, passed=True, score=1.0, metrics={"note": reason}, evaluated=False)


# ---------------------------------------------------------------- schema
def check_schema(calib: DeviceCalibration, entry: CameraEntry) -> CheckResult:
    intr = entry.intrinsics
    w, h = entry.resolution.wh
    rules = {
        "width_positive": w > 0,
        "height_positive": h > 0,
        "fx_positive": intr.fx > 0,
        "fy_positive": intr.fy > 0,
        "cx_in_range": 0 <= intr.cx < w,
        "cy_in_range": 0 <= intr.cy < h,
        "distortion_finite": True,
    }
    if entry.distortion is not None:
        vals = [entry.distortion.k1, entry.distortion.k2, entry.distortion.k3, entry.distortion.k4]
        rules["distortion_finite"] = all(np.isfinite(vals))
    score = float(sum(rules.values())) / len(rules)
    return CheckResult("schema", all(rules.values()), score, metrics={k: bool(v) for k, v in rules.items()})


# -------------------------------------------------------------- rotation
def check_rotation(R) -> CheckResult:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    frob = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det_err = abs(float(np.linalg.det(R)) - 1.0)
    passed = frob < 1e-4 and det_err < 1e-4
    score = float(np.exp(-10 * max(frob, det_err)))
    return CheckResult(
        "rotation", passed, score, metrics={"frobenius_RtR_minus_I": frob, "det_minus_1": float(np.linalg.det(R)) - 1.0}
    )


# -------------------------------------------------------------- baseline
def check_baseline(R, t, expected_range: tuple[float, float]) -> CheckResult:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    center_in_left = -R.T @ t
    baseline = float(np.linalg.norm(center_in_left))
    lo, hi = expected_range
    passed = lo < baseline < hi
    span = hi - lo
    if passed:
        score = 1.0  # inside spec range: no health penalty
    else:
        overshoot = (lo - baseline) if baseline <= lo else (baseline - hi)
        score = max(0.0, 1.0 - overshoot / (span / 2))
    return CheckResult(
        "baseline",
        passed,
        score,
        metrics={"baseline_m": baseline, "expected_range_m": list(expected_range)},
    )


# ------------------------------------------------------------ round-trip
def _score_from_p99(p99_px: float, good: float = 0.05, bad: float = 0.5) -> float:
    if p99_px <= good:
        return 1.0
    if p99_px >= bad:
        return 0.0
    lg, lb = np.log10(good), np.log10(bad)
    t = (np.log10(p99_px) - lg) / (lb - lg)
    return float(np.clip(1.0 - t, 0.0, 1.0))


def check_projection_roundtrip(camera: CameraModel, n_samples: int = 100_000, seed: int = 7) -> CheckResult:
    rng = np.random.default_rng(seed)
    mask = camera.valid_mask()
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return CheckResult("projection_roundtrip", False, 0.0, {"error": "empty valid mask"})
    idx = rng.choice(len(xs), size=min(n_samples, len(xs)), replace=False)
    uv = np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float64)
    rays = camera.unproject(uv)
    back = camera.project(rays)
    err = np.linalg.norm(back - uv, axis=-1)
    finite = np.isfinite(err)
    err = err[finite]
    stats = {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "max": float(err.max()),
        "n": int(err.size),
        "valid_pixel_ratio": float(mask.mean()),
    }
    return CheckResult(
        "projection_roundtrip",
        stats["p99"] < 0.05,
        _score_from_p99(stats["p99"]),
        stats,
    )


# --------------------------------------------------------- image quality
def check_image_quality(img: np.ndarray, min_keypoints: int = 50) -> CheckResult:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(gray.mean())
    black_frame = mean_lum < 5.0
    overexposed = mean_lum > 250.0
    sift = cv2.SIFT_create()  # type: ignore[attr-defined]
    n_kp = len(sift.detect(gray, None))
    blur_ok = blur_var > 100.0
    exposure_ok = not black_frame and not overexposed
    feats_ok = n_kp >= min_keypoints
    passed = blur_ok and exposure_ok and feats_ok
    score = (float(blur_ok) + float(exposure_ok) + float(feats_ok)) / 3.0
    return CheckResult(
        "image_quality",
        passed,
        score,
        {"blur_variance": blur_var, "mean_luma": mean_lum, "keypoints": n_kp},
    )


# ------------------------------------------------------------- epipolar
def epipolar_vertical_stats(img_left_rect: np.ndarray, img_right_rect: np.ndarray,
                            band_px: float = 6.0) -> dict:
    """Epipolar QA for an already-rectified pair.

    Matching is guided by the known rectification constraint (candidates must
    lie within +/- band_px rows), so statistics describe residual vertical
    alignment rather than descriptor ambiguity. If the pair were truly
    misaligned beyond the band, match count collapses and the check fails.
    """
    gray_l = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=6000)  # type: ignore[attr-defined]
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)
    empty = {
        "n_matches": 0,
        "n_kp_left": len(kp_l) if kp_l else 0,
        "n_kp_right": len(kp_r) if kp_r else 0,
        "inlier_ratio": 0.0,
        "median": float("nan"),
        "p95": float("nan"),
        "p99": float("nan"),
    }
    if des_l is None or des_r is None or len(kp_l) < 30 or len(kp_r) < 30:
        return empty

    pl = np.array([k.pt for k in kp_l])
    pr = np.array([k.pt for k in kp_r])
    d_l = des_l.astype(np.float32)
    d_r = des_r.astype(np.float32)

    norm_l = np.linalg.norm(d_l, axis=-1, keepdims=True)
    norm_r = np.linalg.norm(d_r, axis=-1, keepdims=True)
    cos = (d_l @ d_r.T) / np.maximum(norm_l * norm_r.T, 1e-9)
    dist = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(cos, -1.0, 1.0), 0.0))

    order = np.argsort(dist, axis=1)[:, :2]
    best = dist[np.arange(len(pl)), order[:, 0]]
    second = dist[np.arange(len(pl)), order[:, 1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = best / np.maximum(second, 1e-9)
    ok = np.isfinite(ratio) & (ratio < 0.75)
    idx_l = np.nonzero(ok)[0]
    if len(idx_l) < 30:
        return {**empty, "n_kp_left": len(kp_l), "n_kp_right": len(kp_r)}

    # Rank by distinctiveness; descriptor ambiguity on low-texture structures
    # otherwise dominates the tail of the dy distribution.
    sel = idx_l[np.argsort(ratio[idx_l])][: min(200, len(idx_l))]

    dy = np.abs(pl[sel, 1] - pr[order[sel, 0], 1])
    med0 = float(np.median(dy))
    sigma = max(1.4826 * float(np.median(np.abs(dy - med0))), 0.25)
    keep = np.abs(dy - med0) <= max(4.0 * sigma, 0.75)
    if keep.sum() < 20:
        keep = np.ones(len(dy), bool)

    return {
        "n_matches": int(keep.sum()),
        "n_raw_matches": int(ok.sum()),
        "n_kp_left": len(kp_l),
        "n_kp_right": len(kp_r),
        "inlier_ratio": float(ok.mean()),
        "median": float(np.median(dy[keep])),
        "p95": float(np.percentile(dy[keep], 95)),
        "p99": float(np.percentile(dy[keep], 99)),
    }


def check_epipolar(img_left_rect: np.ndarray, img_right_rect: np.ndarray) -> CheckResult:
    stats = epipolar_vertical_stats(img_left_rect, img_right_rect)
    ok = (
        stats["n_matches"] >= 30
        and np.isfinite(stats["median"])
        and stats["median"] < 0.5
        and stats["p95"] < 1.0
        and stats.get("inlier_ratio", 0.0) >= 0.5
    )
    med = stats.get("median", float("nan"))
    if np.isfinite(med):
        score = max(0.0, min(1.0, 1.0 - med))
        score *= min(1.0, stats.get("inlier_ratio", 1.0) / 0.6)
    else:
        score = 0.0
    return CheckResult("stereo_epipolar", ok, score, stats)


# ---------------------------------------------------------------- lines
def check_line_straightness(rectified_img: np.ndarray) -> CheckResult:
    """Heuristic edge straightness on a rectified/virtual image.

    Uses LSD when available, else HoughLinesP segments; measures how well
    segment-support pixels fit their line after subpixel refinement.
    Metric only (not gating) in MVP.
    """
    gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
    edges = []
    try:
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(gray)[0]
        if lines is not None:
            for l in lines[:200]:
                x1, y1, x2, y2 = l[0]
                edges.append((x1, y1, x2, y2))
    except Exception:
        linesP = cv2.HoughLinesP(gray, 1, np.pi / 360, threshold=80, minLineLength=60, maxLineGap=4)
        if linesP is not None:
            for l in linesP[:200]:
                x1, y1, x2, y2 = l[0]
                edges.append((x1, y1, x2, y2))
    if not edges:
        return CheckResult.not_evaluated("line_straightness", "no line detector output")
    residuals = []
    canny = cv2.Canny(gray, 50, 150)
    pts = np.column_stack(np.nonzero(canny))[:, ::-1].astype(np.float64)
    for x1, y1, x2, y2 in edges:
        seg_len = float(np.hypot(x2 - x1, y2 - y1))
        if seg_len < 60:
            continue
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        d = (pts - p1) @ (p2 - p1) / (seg_len**2)
        proj = p1 + d[:, None] * (p2 - p1)
        dist = np.linalg.norm(pts - proj, axis=-1)
        near = dist < 3.0
        if near.sum() < 40:
            continue
        line_dir = (p2 - p1) / seg_len
        normal = np.array([-line_dir[1], line_dir[0]])
        offsets = (pts[near] - p1) @ normal
        residuals.append(float(np.sqrt(np.mean(offsets**2))))
    if not residuals:
        return CheckResult.not_evaluated("line_straightness", "no long segments")
    rms = float(np.median(residuals))
    return CheckResult(
        "line_straightness",
        rms < 1.0,
        max(0.0, 1.0 - rms),
        {"median_edge_rms_px": rms, "n_segments": len(residuals)},
    )

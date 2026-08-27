"""Nonlinear camera model fitting from pixel<->ray observations.

Powers the AnyCalib fallback (fit any supported model to dense predicted
rays) and model selection. Uses Levenberg-Marquardt with numeric Jacobian;
no scipy/Ceres dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.base import CameraModel
from ..models.brown import BrownConrady
from ..models.double_sphere import DoubleSphere
from ..models.eucm import EUCM
from ..models.kb4 import KannalaBrandt4
from ..models.pinhole import PinholeCamera

MODEL_TYPES = ("kannala_brandt_4", "eucm", "double_sphere", "pinhole", "brown_conrady")

PARAM_NAMES: dict[str, list[str]] = {
    "kannala_brandt_4": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
    "eucm": ["fx", "fy", "cx", "cy", "alpha"],
    "double_sphere": ["fx", "fy", "cx", "cy", "xi", "alpha"],
    "pinhole": ["fx", "fy", "cx", "cy"],
    "brown_conrady": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"],
}

PARAM_SCALES: dict[str, np.ndarray] = {
    "kannala_brandt_4": np.array([50, 50, 50, 50, 0.05, 0.01, 0.005, 0.005]),
    "eucm": np.array([50, 50, 50, 50, 0.1]),
    "double_sphere": np.array([50, 50, 50, 50, 0.1, 0.1]),
    "pinhole": np.array([50, 50, 50, 50]),
    "brown_conrady": np.array([50, 50, 50, 50, 0.05, 0.01, 0.005, 0.01, 0.01]),
}

MODEL_CLASSES: dict[str, Any] = {
    "brown_conrady": BrownConrady,
    "kannala_brandt_4": KannalaBrandt4,
    "eucm": EUCM,
    "double_sphere": DoubleSphere,
    "pinhole": PinholeCamera,
}


def make_camera(model_type: str, theta: np.ndarray, width: int, height: int) -> CameraModel:
    names = PARAM_NAMES[model_type]
    p = dict(zip(names, [float(v) for v in theta]))
    cls = MODEL_CLASSES[model_type]
    if model_type == "kannala_brandt_4":
        return cls(p["fx"], p["fy"], p["cx"], p["cy"], p["k1"], p["k2"], p["k3"], p["k4"], width, height)
    if model_type == "eucm":
        return cls(p["fx"], p["fy"], p["cx"], p["cy"], p["alpha"], width, height)
    if model_type == "double_sphere":
        return cls(p["fx"], p["fy"], p["cx"], p["cy"], p["xi"], p["alpha"], width, height)
    if model_type == "brown_conrady":
        return cls(p["fx"], p["fy"], p["cx"], p["cy"], p["k1"], p["k2"], p["k3"],
                   p["p1"], p["p2"], width, height)
    return cls(p["fx"], p["fy"], p["cx"], p["cy"], width, height)


def initial_theta(model_type: str, width: int, height: int, prior: CameraModel | None = None) -> np.ndarray:
    if prior is not None and prior.model_type == model_type:
        d = prior.intrinsics_dict()
        extra = prior.distortion_dict() or {}
        return np.array([d.get(n, extra.get(n, 0.0)) for n in PARAM_NAMES[model_type]], dtype=np.float64)
    f0 = 0.6 * width
    base = {"fx": f0, "fy": f0, "cx": (width - 1) / 2, "cy": (height - 1) / 2}
    defaults = {"alpha": 0.5, "xi": 0.5, "k1": 0.0, "k2": 0.0, "k3": 0.0,
                "k4": 0.0, "p1": 0.0, "p2": 0.0}
    return np.array([base.get(n, defaults.get(n, 0.0)) for n in PARAM_NAMES[model_type]], dtype=np.float64)


def _clip_theta(model_type: str, theta: np.ndarray) -> np.ndarray:
    """Keep LM iterates inside the model's valid parameter domain."""
    t = theta.copy()
    t[0] = np.clip(t[0], 1e-3, None)
    t[1] = np.clip(t[1], 1e-3, None)
    if model_type == "eucm":
        t[4] = np.clip(t[4], 0.0, 0.98)
    elif model_type == "double_sphere":
        t[4] = np.clip(t[4], -2.0, 4.0)
        t[5] = np.clip(t[5], 0.0, 1.0)
    elif model_type == "kannala_brandt_4":
        t[4:] = np.clip(t[4:], -1.5, 1.5)
    return t


def _numeric_jacobian(fn, x: np.ndarray, r0: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    jac = np.empty((len(r0), len(x)))
    for i in range(len(x)):
        step = max(1e-6, rel_step * abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += step
        xm[i] -= step
        jac[:, i] = (fn(xp) - fn(xm)) / (2.0 * step)
    return jac


def levenberg_marquardt(residual_fn, x0: np.ndarray, max_iters: int = 120, lam0: float = 1e-3, jac_rel_step: float = 1e-4):
    x = np.asarray(x0, dtype=np.float64).copy()
    r = residual_fn(x)
    cost = float(r @ r)
    lam = lam0
    for _ in range(max_iters):
        J = _numeric_jacobian(residual_fn, x, r, rel_step=jac_rel_step)
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(15):
            try:
                dx = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x + dx
            rn = residual_fn(xn)
            cn = float(rn @ rn)
            if np.isfinite(cn) and cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            lam *= 10
        if not improved:
            break
    return x, cost


@dataclass
class FitResult:
    model_type: str
    theta: np.ndarray
    median_ray_err_deg: float
    p99_ray_err_deg: float
    rmse_px: float
    n_points: int
    params: dict

    def to_json(self) -> dict:
        return {
            "model_type": self.model_type,
            "params": self.params,
            "median_ray_err_deg": self.median_ray_err_deg,
            "p99_ray_err_deg": self.p99_ray_err_deg,
            "rmse_px": self.rmse_px,
            "n_points": self.n_points,
        }


def fit_model(
    model_type: str,
    pixels: np.ndarray,
    rays: np.ndarray,
    width: int,
    height: int,
    init: CameraModel | None = None,
    max_points: int = 2000,
    seed: int = 0,
) -> FitResult:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    if len(pixels) > max_points:
        idx = np.random.default_rng(seed).choice(len(pixels), max_points, replace=False)
        pixels, rays = pixels[idx], rays[idx]
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def residuals(theta: np.ndarray) -> np.ndarray:
        try:
            cam = make_camera(model_type, _clip_theta(model_type, theta), width, height)
            pred = cam.unproject(pixels)
        except Exception:
            return np.full(3 * len(pixels), 1e3)
        bad = ~np.isfinite(pred)
        if np.any(bad):
            pred = np.where(bad, 1e3, pred)
        return (pred - rays).ravel()

    theta0 = initial_theta(model_type, width, height, prior=init)
    if model_type in ("eucm", "double_sphere") and init is None:
        # Pinhole unprojection is linear and always converges; seed its
        # intrinsics into the richer model to avoid bad local minima.
        pin = fit_model("pinhole", pixels, rays, width, height, init=None,
                        max_points=max_points)
        theta0 = theta0.copy()
        theta0[:4] = pin.theta[:4]
    theta, _ = levenberg_marquardt(residuals, theta0)
    theta = _clip_theta(model_type, theta)
    cam = make_camera(model_type, theta, width, height)
    pred = cam.unproject(pixels)
    with np.errstate(invalid="ignore"):
        ang = np.degrees(np.arccos(np.clip(np.sum(pred * rays, axis=-1), -1, 1)))
        uv_back = cam.project(rays)
        px_err = np.sum((uv_back - pixels) ** 2, axis=-1)
    ang = ang[np.isfinite(ang)]
    px_err = px_err[np.isfinite(px_err)]
    rmse = float(np.sqrt(np.mean(px_err))) if len(px_err) else 1e9
    return FitResult(
        model_type=model_type,
        theta=theta,
        median_ray_err_deg=float(np.median(ang)) if len(ang) else 1e6,
        p99_ray_err_deg=float(np.percentile(ang, 99)) if len(ang) else 1e6,
        rmse_px=rmse,
        n_points=len(pixels),
        params={n: float(v) for n, v in zip(PARAM_NAMES[model_type], theta)},
    )


def select_best_model(results: list[FitResult], w_ray: float = 0.6, w_px: float = 0.2, w_complexity: float = 0.2) -> FitResult:
    """Spec section 36: lowest combined score wins.

    Errors are log-normalized so a candidate that genuinely fails (orders of
    magnitude worse) dominates the score; parameter-count complexity only
    breaks near-ties between models that fit equally well.
    """
    if not results:
        raise ValueError("no fit results")

    def norm_log(vals, floor=1e-6):
        v = np.maximum(np.asarray(vals, dtype=np.float64), floor)
        lg = np.log10(v)
        rng = float(lg.max() - lg.min())
        return (lg - lg.min()) / (rng + 1e-12)

    def norm(vals):
        v = np.asarray(vals, dtype=np.float64)
        rng = float(v.max() - v.min())
        return (v - v.min()) / (rng + 1e-12)

    e_ray = norm_log([r.median_ray_err_deg for r in results])
    e_px = norm_log([max(r.rmse_px, 1e-3) for r in results])
    comp = norm([len(PARAM_NAMES[r.model_type]) for r in results])
    scores = w_ray * e_ray + w_px * e_px + w_complexity * comp
    return results[int(np.argmin(scores))]

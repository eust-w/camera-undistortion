"""AnyCalib-style fallback: fit camera models to dense pixel->ray predictions.

The reference AnyCalib uses a network that predicts a view ray for (almost)
every pixel. This module implements the surrounding machinery so any dense
ray predictor can plug in:

  * per-frame robust model fitting (KB4 / EUCM / Double Sphere / pinhole)
  * multi-frame consensus (elementwise median + refit)
  * automatic model selection (spec section 36)

MVP ships a track-based provider (`TrackRayProvider`): rays estimated from
multi-view feature tracks (see colmap_refine.collect_track_observations), so
the full fallback chain works without any neural dependency. To use the
actual AnyCalib network, wrap it in a callable matching `DenseRayProvider`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..calibration.fitting import FitResult, fit_model, select_best_model
from ..models.base import CameraModel
from ..schema import CameraEntry, Intrinsics, ModelSpec, Resolution

DenseRayProvider = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray] | None]

DEFAULT_CANDIDATES = ("kannala_brandt_4", "eucm", "double_sphere", "pinhole")


@dataclass
class AnyCalibResult:
    best: FitResult
    all_results: list[FitResult] = field(default_factory=list)
    n_frames: int = 0
    n_points: int = 0

    def to_json(self) -> dict:
        return {
            "best": self.best.to_json(),
            "all": [r.to_json() for r in self.all_results],
            "n_frames": self.n_frames,
            "n_points": self.n_points,
        }


def fit_from_observations(
    observations: Sequence[tuple[np.ndarray, np.ndarray]],
    width: int,
    height: int,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
    init: CameraModel | None = None,
    max_points_per_frame: int = 1500,
) -> AnyCalibResult:
    """Fit every candidate model to every frame, then consensus + selection."""
    if not observations:
        raise ValueError("no observations")
    per_model: dict[str, list[FitResult]] = {m: [] for m in candidates}
    n_points = 0
    for pixels, rays in observations:
        if len(pixels) < 100:
            continue
        n_points += len(pixels)
        for model_type in candidates:
            per_model[model_type].append(fit_model(model_type, pixels, rays, width, height, init=init,
                                                   max_points=max_points_per_frame))

    consensus: list[FitResult] = []
    all_results: list[FitResult] = []
    pixels_all = np.concatenate([p for p, _ in observations])
    rays_all = np.concatenate([r for _, r in observations])
    for model_type, fits in per_model.items():
        if not fits:
            continue
        all_results.extend(fits)
        # consensus: elementwise median of per-frame params, then LM polish
        # on the pooled observations starting from that median (more stable
        # than refitting from a generic init).
        theta_median = np.median(np.stack([f.theta for f in fits]), axis=0)
        from ..calibration.fitting import make_camera

        init_cam = make_camera(model_type, theta_median, width, height)
        consensus.append(fit_model(model_type, pixels_all, rays_all, width, height,
                                   init=init_cam, max_points=4000))
    if not consensus:
        raise ValueError("not enough observations to fit any model")
    best = select_best_model(consensus)
    return AnyCalibResult(best=best, all_results=all_results,
                          n_frames=len(observations), n_points=n_points)


def anycalib_fit(
    provider_or_observations: DenseRayProvider | Sequence[tuple[np.ndarray, np.ndarray]],
    images: Sequence[np.ndarray] | None = None,
    width: int = 1600,
    height: int = 1300,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
) -> AnyCalibResult:
    """Accept either precomputed (pixels, rays) observations or a provider + images."""
    if callable(provider_or_observations) and images is not None:
        observations = []
        for img in images:
            out = provider_or_observations(img)
            if out is not None:
                observations.append(out)
        return fit_from_observations(observations, width, height, candidates)
    return fit_from_observations(list(provider_or_observations), width, height, candidates)


def build_camera_entry(result: AnyCalibResult, camera_id: str, width: int, height: int) -> CameraEntry:
    """Wrap a fitted model into a canonical CameraEntry (source: anycalib)."""
    params = dict(result.best.params)
    intr = {k: params.pop(k) for k in ("fx", "fy", "cx", "cy")}
    model_type = result.best.model_type
    extra = {}
    if model_type == "kannala_brandt_4":
        from ..schema import KB4Distortion

        distortion = KB4Distortion(**{k: params.pop(k) for k in ("k1", "k2", "k3", "k4")})
    else:
        distortion = None
        extra = params
    return CameraEntry(
        id=camera_id,
        resolution=Resolution(width=width, height=height),
        model=ModelSpec(type=model_type, convention="anycalib_v1", params=extra),
        intrinsics=Intrinsics(**intr),
        distortion=distortion,
    )

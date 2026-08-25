"""Fallback stages for the calibration state machine (spec sections 33-36).

Chain: Factory QA fail -> stereo prior refinement -> AnyCalib -> QUARANTINE.
"""

from __future__ import annotations

from .anycalib import (
    AnyCalibResult,
    DenseRayProvider,
    anycalib_fit,
    build_camera_entry,
    fit_from_observations,
)
from .colmap_refine import (
    PairObservation,
    StereoPairResult,
    StereoRefineReport,
    refine_from_stereo_frames,
    refine_stereo_pair,
)

__all__ = [
    "AnyCalibResult",
    "DenseRayProvider",
    "PairObservation",
    "StereoPairResult",
    "StereoRefineReport",
    "anycalib_fit",
    "build_camera_entry",
    "fit_from_observations",
    "refine_from_stereo_frames",
    "refine_stereo_pair",
]

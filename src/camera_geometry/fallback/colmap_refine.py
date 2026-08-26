"""Factory-calibration refinement from stereo frames (spec section 34).

Identifiability note (verified experimentally, see tests):
  * Monocular epipolar self-consistency CANNOT identify intrinsics — a free
    essential matrix absorbs smooth distortion errors.
  * A STEREO rig with factory-known extrinsics has a FIXED essential matrix
    E = [t]x R. Wrong undistortion then produces large, monotonic fixed-E
    Sampson error with a sharp minimum at the true calibration.

Objective (joint over both cameras):

  L(th_L, th_R) = mean_i sqrt( sampson(n_L(th_L; u_L^i), n_R(th_R; u_R^i); E_fixed) )
                + lambda * (prior regularization on both parameter vectors)

Monocular baseline *scale* is never refined (unobservable); baseline
plausibility is enforced by baseline QA.

An external COLMAP binary can replace this stage later; the interface
(`refine_from_stereo_frames -> StereoRefineReport`) is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..calibration.fitting import (
    PARAM_NAMES,
    PARAM_SCALES,
    _clip_theta,
    levenberg_marquardt,
    make_camera,
)
from ..models.base import CameraModel
from ..schema import CameraEntry
from ..transforms import SE3


@dataclass
class PairObservation:
    camera_id: str
    pixels: np.ndarray  # [N,2]
    rays: np.ndarray    # [N,3] unit rays under refined model


@dataclass
class StereoPairResult:
    theta_left: np.ndarray
    theta_right: np.ndarray
    err_before_px: float
    err_after_px: float
    n_points: int
    observations: PairObservation | None = None


@dataclass
class StereoRefineReport:
    entries: dict[str, CameraEntry]
    theta_left: np.ndarray
    theta_right: np.ndarray
    err_before_px: float
    err_after_px: float
    n_pairs: int
    n_points: int
    observations: list[PairObservation] = field(default_factory=list)
    per_pair: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "left": {n: float(v) for n, v in zip(PARAM_NAMES["kannala_brandt_4"], self.theta_left)},
            "right": {n: float(v) for n, v in zip(PARAM_NAMES["kannala_brandt_4"], self.theta_right)},
            "err_before_px": self.err_before_px,
            "err_after_px": self.err_after_px,
            "n_pairs": self.n_pairs,
            "n_points": self.n_points,
        }


def _theta_from_camera(entry: CameraEntry, cam: CameraModel) -> np.ndarray:
    names = PARAM_NAMES[entry.model.type]
    d = cam.intrinsics_dict()
    extra = cam.distortion_dict() or {}
    return np.array([d.get(n, extra.get(n, 0.0)) for n in names], dtype=np.float64)


_SUBPIX = True  # subpixel corner refinement on matched keypoints


def _match_stereo(img_l, img_r, min_matches: int = 100):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=6000)  # type: ignore[attr-defined]
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)
    if des_l is None or des_r is None:
        return None
    matches = cv2.BFMatcher().knnMatch(des_l, des_r, k=2)
    pts_l, pts_r = [], []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            pts_l.append(kp_l[m.queryIdx].pt)
            pts_r.append(kp_r[m.trainIdx].pt)
    if len(pts_l) < min_matches:
        return None
    uv1 = np.asarray(pts_l, dtype=np.float32).reshape(-1, 1, 2)
    uv2 = np.asarray(pts_r, dtype=np.float32).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    if _SUBPIX:
        try:
            cv2.cornerSubPix(gray_l, uv1, (5, 5), (-1, -1), crit)
            cv2.cornerSubPix(gray_r, uv2, (5, 5), (-1, -1), crit)
        except cv2.error:
            pass
    return uv1.reshape(-1, 2).astype(np.float64), uv2.reshape(-1, 2).astype(np.float64)


def _fixed_essential(T_left_right: SE3) -> np.ndarray:
    t = T_left_right.t
    Tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    return Tx @ T_left_right.R


def _normalized(cam: CameraModel, uv: np.ndarray) -> np.ndarray:
    r = np.asarray(cam.unproject(uv))
    z = r[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, np.nan, z)
    return np.concatenate([r[:, :2] / z, np.ones((len(r), 1))], axis=1)


def _sampson(n1: np.ndarray, n2: np.ndarray, E: np.ndarray) -> np.ndarray:
    E_n1 = n1 @ E.T
    E_t_n2 = n2 @ E
    num = np.sum(n2 * E_n1, axis=-1) ** 2
    den = E_n1[:, 0] ** 2 + E_n1[:, 1] ** 2 + E_t_n2[:, 0] ** 2 + E_t_n2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def _identifiable_mask(model_type: str, T_left_right: SE3) -> np.ndarray:
    """Parameters observable through the fixed-E stereo constraint.

    For a predominantly horizontal baseline the essential-matrix constraint is
    vertical-parity only: fx/cx never enter it and must stay anchored at the
    prior. Radial + vertical params are identifiable.
    """
    names = PARAM_NAMES[model_type]
    t = np.abs(T_left_right.t)
    horizontal = t[0] >= t[1] and t[0] >= t[2]
    if not horizontal:
        return np.ones(len(names), dtype=bool)
    # fy/cy/k1 cover the dominant factory error modes; higher-order k's stay
    # anchored (freeing them invites oscillatory overfit on sparse signal)
    vertical_only = {"fx": False, "fy": True, "cx": False, "cy": True,
                     "k1": True, "k2": False, "k3": False, "k4": False,
                     "alpha": True, "xi": True}
    return np.array([vertical_only.get(n, True) for n in names], dtype=bool)


def refine_stereo_pair(
    img_l: np.ndarray,
    img_r: np.ndarray,
    entry_l: CameraEntry,
    entry_r: CameraEntry,
    cam_l: CameraModel,
    cam_r: CameraModel,
    T_left_right: SE3,
    lambda_reg: float = 0.05,
    max_points: int = 2500,
    inlier_thresh_px: float = 25.0,
    reference_camera_id: str | None = None,
) -> StereoPairResult | None:
    matched = _match_stereo(img_l, img_r)
    if matched is None:
        return None
    uv_l, uv_r = matched

    E_fix = _fixed_essential(T_left_right)
    err0 = _sampson(_normalized(cam_l, uv_l), _normalized(cam_r, uv_r), E_fix)
    keep = err0 < inlier_thresh_px
    uv_l, uv_r = uv_l[keep], uv_r[keep]
    if len(uv_l) < 100:
        return None
    if len(uv_l) > max_points:
        idx = np.random.default_rng(0).choice(len(uv_l), max_points, replace=False)
        uv_l, uv_r = uv_l[idx], uv_r[idx]

    theta_l0 = _theta_from_camera(entry_l, cam_l)
    theta_r0 = _theta_from_camera(entry_r, cam_r)
    theta0 = np.concatenate([theta_l0, theta_r0])
    scales = np.concatenate([PARAM_SCALES[entry_l.model.type], PARAM_SCALES[entry_r.model.type]])
    mt = entry_l.model.type
    mr = entry_r.model.type
    nl = len(theta_l0)
    mask_l = _identifiable_mask(mt, T_left_right)
    mask_r = _identifiable_mask(mr, T_left_right)

    def _signed_sampson(thetas: np.ndarray) -> np.ndarray:
        """SIGNED sampson residuals: the sign carries the downhill direction,
        which unsigned |sampson| destroys (source of bad local minima)."""
        try:
            cl = make_camera(mt, _clip_theta(mt, thetas[:nl]), cam_l.width, cam_l.height)
            cr = make_camera(mr, _clip_theta(mr, thetas[nl:]), cam_r.width, cam_r.height)
            n_l = _normalized(cl, uv_l)
            n_r = _normalized(cr, uv_r)
        except Exception:
            return np.full(len(uv_l), 1e6)
        E_n1 = n_l @ E_fix.T
        E_t_n2 = n_r @ E_fix
        num = np.sum(n_r * E_n1, axis=-1)
        den = E_n1[:, 0] ** 2 + E_n1[:, 1] ** 2 + E_t_n2[:, 0] ** 2 + E_t_n2[:, 1] ** 2
        r = num / np.maximum(den, 1e-12)
        return np.where(np.isfinite(r), r, 1e6)

    def _huber(r: np.ndarray, delta: float = 2.0) -> np.ndarray:
        a = np.abs(r)
        return np.where(a <= delta, r, np.sign(r) * np.sqrt(np.maximum(delta * (2.0 * a - delta), 1e-12)))

    def _err(thetas: np.ndarray) -> np.ndarray:
        return np.abs(_huber(_signed_sampson(thetas)))

    def _reg_cost(theta_full: np.ndarray) -> float:
        return lambda_reg * float(np.sum(((theta_full - theta0) / scales) ** 2))

    best_theta = theta0.copy()
    best_cost: float = float(np.sum(_err(theta0) ** 2)) + _reg_cost(theta0)

    def _golden(coord: int, lo: float, hi: float, theta_full: np.ndarray, iters: int = 40) -> float:
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        def f(v):
            th = theta_full.copy()
            th[coord] = v
            return float(np.sum(_err(th) ** 2)) + _reg_cost(th)
        f_cur = f(theta_full[coord])
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = f(d)
        v = (a + b) / 2.0
        # never accept a move worse than the current value (noise guard)
        return v if f(v) < f_cur else theta_full[coord]

    # One-sided coordinate descent: the reference camera is the calibration
    # anchor and stays frozen. (With both sides free, "fix R" and "equivalently
    # warp L" are indistinguishable from parity alone.)
    sides = []
    if reference_camera_id != entry_r.id:
        sides.append(("right", np.nonzero(mask_r)[0] + nl))
    if reference_camera_id != entry_l.id:
        sides.append(("left", np.nonzero(mask_l)[0]))
    if not sides:
        return None
    for side_free, free_idx in sides:
        if len(free_idx) == 0:
            continue
        theta_side = theta0.copy()
        cost_side = None
        for _sweep in range(25):
            for coord in free_idx:
                span = max(3.0 * scales[coord], 0.5 * abs(theta_side[coord] - theta0[coord]) + 1e-9)
                lo = theta_side[coord] - span
                hi = theta_side[coord] + span
                theta_side[coord] = _golden(int(coord), lo, hi, theta_side)
            cost_side = float(np.sum(_err(theta_side) ** 2)) + _reg_cost(theta_side)
        # joint LM polish from the coordinate-descent point
        def _res_joint(theta_free: np.ndarray) -> np.ndarray:
            theta_full = theta0.copy()
            theta_full[free_idx] = theta_free
            s = _huber(_signed_sampson(theta_full))
            reg = np.sqrt(lambda_reg) * (theta_full - theta0) / scales
            return np.concatenate([s, reg[free_idx]])

        try:
            theta_pol, _ = levenberg_marquardt(
                _res_joint, theta_side[free_idx].copy(), max_iters=80, jac_rel_step=2e-3
            )
            theta_full = theta0.copy()
            theta_full[free_idx] = theta_pol
            cost_pol = float(np.sum(_err(theta_full) ** 2)) + _reg_cost(theta_full)
            if cost_side is not None and cost_pol < cost_side:
                theta_side = theta_full
                cost_side = cost_pol
        except Exception:
            pass
        cost_side = float(np.sum(_err(theta_side) ** 2)) + _reg_cost(theta_side)
        if cost_side < best_cost:
            best_cost = cost_side
            best_theta = theta_side

    theta = best_theta
    theta[:nl] = _clip_theta(mt, theta[:nl])
    theta[nl:] = _clip_theta(mr, theta[nl:])

    err_before = float(np.mean(_err(theta0)))
    theta = best_theta
    err_after = float(np.mean(_err(theta)))

    return StereoPairResult(
        theta_left=theta[:nl],
        theta_right=theta[nl:],
        err_before_px=err_before,
        err_after_px=err_after,
        n_points=len(uv_l),
    )


def _apply_theta(entry: CameraEntry, cam_new: CameraModel) -> CameraEntry:
    e = entry.model_copy(deep=True)
    e.intrinsics = e.intrinsics.model_copy(update=cam_new.intrinsics_dict())
    dist = cam_new.distortion_dict()
    if entry.model.type == "kannala_brandt_4" and dist is not None:
        if e.distortion is None:
            from ..schema import KB4Distortion

            e.distortion = KB4Distortion(**dist)
        else:
            e.distortion = e.distortion.model_copy(update=dist)
    elif dist is not None:
        e.model.params = {**(e.model.params or {}), **dist}
    return e


def refine_stereo_pooled(
    stereo_images: list[tuple[np.ndarray, np.ndarray]],
    entry_l: CameraEntry,
    entry_r: CameraEntry,
    cam_l: CameraModel,
    cam_r: CameraModel,
    T_left_right: SE3,
    reference_camera_id: str | None = None,
    lambda_reg: float = 0.05,
    max_points_per_pair: int = 600,
    inlier_thresh_px: float = 25.0,
) -> StereoPairResult | None:
    """Pool ALL pairs into one LM problem with shared target parameters.

    Per-pair fits vary by several fy-units (noise); pooling N pairs shrinks
    the variance ~sqrt(N) and is the preferred stage-1 estimator.
    """
    E_fix = _fixed_essential(T_left_right)
    mt, mr = entry_l.model.type, entry_r.model.type
    nl = len(PARAM_NAMES[mt])

    theta_l0 = _theta_from_camera(entry_l, cam_l)
    theta_r0 = _theta_from_camera(entry_r, cam_r)
    theta0 = np.concatenate([theta_l0, theta_r0])
    scales = np.concatenate([PARAM_SCALES[mt], PARAM_SCALES[mr]])

    if reference_camera_id == entry_r.id:
        mask = _identifiable_mask(mt, T_left_right)
        off = 0
    else:
        mask = _identifiable_mask(mr, T_left_right)
        off = nl
    free_local = np.nonzero(mask)[0]
    if len(free_local) == 0:
        return None
    free_idx = free_local + off

    matches = []
    for il, ir in stereo_images:
        m = _match_stereo(il, ir)
        if m is None:
            continue
        uvl, uvr = m
        s = _sampson(_normalized(cam_l, uvl), _normalized(cam_r, uvr), E_fix)
        keep = np.maximum(np.nan_to_num(s, nan=1e6), 0) < inlier_thresh_px
        if keep.sum() < 100:
            continue
        u1, u2 = uvl[keep], uvr[keep]
        if len(u1) > max_points_per_pair:
            idx = np.random.default_rng(0).choice(len(u1), max_points_per_pair, replace=False)
            u1, u2 = u1[idx], u2[idx]
        matches.append((u1, u2))
    if not matches:
        return None

    n_per = [len(u1) for u1, _ in matches]
    n_total = sum(n_per)

    def _full(theta_free: np.ndarray) -> np.ndarray:
        th = theta0.copy()
        th[free_idx] = theta_free
        return th

    def _signed_all(theta_free: np.ndarray) -> np.ndarray:
        th = _full(theta_free)
        try:
            cl = make_camera(mt, _clip_theta(mt, th[:nl]), cam_l.width, cam_l.height)
            cr = make_camera(mr, _clip_theta(mr, th[nl:]), cam_r.width, cam_r.height)
        except Exception:
            return np.full(n_total, 1e6)
        out = []
        for u1, u2 in matches:
            r_l = np.asarray(cl.unproject(u1))
            r_r = np.asarray(cr.unproject(u2))
            z_l, z_r = r_l[:, 2:3], r_r[:, 2:3]
            z_l = np.where(np.abs(z_l) < 1e-9, np.nan, z_l)
            z_r = np.where(np.abs(z_r) < 1e-9, np.nan, z_r)
            n_l = np.concatenate([r_l[:, :2] / z_l, np.ones((len(r_l), 1))], axis=1)
            n_r = np.concatenate([r_r[:, :2] / z_r, np.ones((len(r_r), 1))], axis=1)
            E_n1 = n_l @ E_fix.T
            E_t_n2 = n_r @ E_fix
            num = np.sum(n_r * E_n1, axis=-1)
            den = E_n1[:, 0] ** 2 + E_n1[:, 1] ** 2 + E_t_n2[:, 0] ** 2 + E_t_n2[:, 1] ** 2
            rr = num / np.maximum(den, 1e-12)
            out.append(np.where(np.isfinite(rr), rr, 1e6))
        return np.concatenate(out)

    def _err_abs(theta_free: np.ndarray) -> np.ndarray:
        r = _signed_all(theta_free)
        a = np.abs(r)
        return np.where(a <= 2.0, a, np.sqrt(np.maximum(2.0 * a - 2.0, 1e-12)))

    def residuals(theta_free: np.ndarray) -> np.ndarray:
        s = _err_abs(theta_free)
        reg = np.sqrt(lambda_reg) * (_full(theta_free) - theta0) / scales
        return np.concatenate([s, reg[free_idx]])

    cost_before = float(np.sum(_err_abs(theta0[free_idx]) ** 2))
    theta_free, _ = levenberg_marquardt(residuals, theta0[free_idx].copy(),
                                        max_iters=80, jac_rel_step=2e-3)
    theta = _full(theta_free)
    theta[:nl] = _clip_theta(mt, theta[:nl])
    theta[nl:] = _clip_theta(mr, theta[nl:])
    cost_after = float(np.sum(_err_abs(theta[free_idx]) ** 2))

    if off:  # right camera refined
        target_entry, target_theta, cam_tgt = entry_r, theta[nl:], cam_r
    else:
        target_entry, target_theta, cam_tgt = entry_l, theta[:nl], cam_l
    cam_new = make_camera(target_entry.model.type, target_theta, cam_tgt.width, cam_tgt.height)
    uv_all = np.concatenate([u for u, _ in matches])
    rays_all = np.asarray(cam_new.unproject(uv_all))
    rays_all /= np.maximum(np.linalg.norm(rays_all, axis=-1, keepdims=True), 1e-12)
    return StereoPairResult(
        theta_left=theta[:nl],
        theta_right=theta[nl:],
        err_before_px=cost_before,
        err_after_px=cost_after,
        n_points=n_total,
        observations=PairObservation(
            camera_id=target_entry.id,
            pixels=uv_all,
            rays=rays_all,
        ),
    )


def refine_from_stereo_frames(
    stereo_images: list[tuple[np.ndarray, np.ndarray]],
    entry_l: CameraEntry,
    entry_r: CameraEntry,
    cam_l_prior: CameraModel,
    cam_r_prior: CameraModel,
    T_left_right: SE3,
    max_pairs: int = 3,
    lambda_reg: float = 0.05,
    collect_obs: bool = True,
    reference_camera_id: str | None = None,
) -> StereoRefineReport | None:
    if len(stereo_images) < 1:
        return None
    n = min(max_pairs, len(stereo_images))
    idx = np.linspace(0, len(stereo_images) - 1, n).astype(int)
    results = []
    for i in idx:
        il, ir = stereo_images[i]
        res = refine_stereo_pair(il, ir, entry_l, entry_r, cam_l_prior, cam_r_prior,
                                 T_left_right, lambda_reg=lambda_reg,
                                 reference_camera_id=reference_camera_id)
        if res is not None and res.err_after_px <= res.err_before_px:
            results.append(res)
    if not results:
        return None

    theta_l = np.median(np.stack([r.theta_left for r in results]), axis=0)
    theta_r = np.median(np.stack([r.theta_right for r in results]), axis=0)
    cam_l_new = make_camera(entry_l.model.type, theta_l, cam_l_prior.width, cam_l_prior.height)
    cam_r_new = make_camera(entry_r.model.type, theta_r, cam_r_prior.width, cam_r_prior.height)

    observations: list[PairObservation] = []
    if collect_obs and stereo_images:
        il, ir = stereo_images[len(stereo_images) // 2]
        matched = _match_stereo(il, ir)
        if matched is not None:
            uv_l, uv_r = matched
            observations = [
                PairObservation(entry_l.id, uv_l,
                                np.asarray(cam_l_new.unproject(uv_l))
                                / np.maximum(np.linalg.norm(
                                    np.asarray(cam_l_new.unproject(uv_l)), axis=-1, keepdims=True), 1e-12)),
                PairObservation(entry_r.id, uv_r,
                                np.asarray(cam_r_new.unproject(uv_r))
                                / np.maximum(np.linalg.norm(
                                    np.asarray(cam_r_new.unproject(uv_r)), axis=-1, keepdims=True), 1e-12)),
            ]

    return StereoRefineReport(
        entries={
            entry_l.id: _apply_theta(entry_l, cam_l_new),
            entry_r.id: _apply_theta(entry_r, cam_r_new),
        },
        theta_left=theta_l,
        theta_right=theta_r,
        err_before_px=float(np.median([r.err_before_px for r in results])),
        err_after_px=float(np.median([r.err_after_px for r in results])),
        n_pairs=len(results),
        n_points=int(sum(r.n_points for r in results)),
        observations=observations,
        per_pair=[{"err_before": r.err_before_px, "err_after": r.err_after_px, "n": r.n_points}
                  for r in results],
    )

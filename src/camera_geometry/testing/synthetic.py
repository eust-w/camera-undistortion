"""Synthetic KB4 stereo scene for demos / golden / integration tests.

Rendering is EXACT w.r.t. the camera model: every pixel is unprojected with
the KB model (via a cached ray map) and intersected with the wall planes, so
images obey epipolar geometry to sub-pixel level regardless of FOV.
"""

from __future__ import annotations

import numpy as np

from ..models.kb4 import KannalaBrandt4
from ..transforms import SE3, roty


def make_kb4_camera(seed: int, width: int, height: int, fx_at_1600: float = 504.9) -> KannalaBrandt4:
    rng = np.random.default_rng(seed)
    fx = fx_at_1600 * width / 1600.0
    return KannalaBrandt4(
        fx * (1 + rng.uniform(-0.003, 0.003)),
        fx * (1 + rng.uniform(-0.003, 0.003)),
        width / 2 - 0.5 + rng.uniform(-6, 6),
        height / 2 - 0.5 + rng.uniform(-4, 4),
        0.072 + rng.uniform(-0.004, 0.004),
        -0.0117 + rng.uniform(-0.001, 0.001),
        0.00034,
        0.00027,
        width,
        height,
    )


def make_stereo_rig(width=1600, height=1300, baseline_m=0.12086, vergence_deg=0.0, seed=11):
    """T_left_right: P_L = R @ P_R + t; right center sits at +x in left frame.

    Parallel optical axes by default (typical stereo rig); pass vergence_deg
    to exercise rotated rigs (covered by analytic tests).
    """
    cam_l = make_kb4_camera(seed, width, height)
    cam_r = make_kb4_camera(seed + 1, width, height)
    R = roty(np.radians(vergence_deg))
    T_left_right = SE3.from_Rt(R, np.array([baseline_m, 0.0, 0.0]))
    return cam_l, cam_r, T_left_right


# ------------------------------------------------------------------ scene
def wall_planes():
    """Walls as (point, unit_normal, axis_s, axis_t) rectangles in room frame."""
    def plane(p0, p1, p2):
        p0, p1, p2 = map(np.asarray, (p0, p1, p2))
        e_s = p1 - p0
        e_t = p2 - p0
        n = np.cross(e_s, e_t)
        n = n / np.linalg.norm(n)
        return {
            "origin": p0,
            "normal": n,
            "e_s": e_s,
            "e_t": e_t,
            "len_s": float(np.linalg.norm(e_s)),
            "len_t": float(np.linalg.norm(e_t)),
            "depth": float(np.mean([p0[2], p1[2], p2[2]])),
        }

    return [
        plane([-4, -2.0, 6], [4, -2.0, 6], [-4, 2.6, 6]),       # front wall
        plane([-4, 1.6, -4], [4, 1.6, -4], [-4, 1.6, 6]),       # floor
        plane([-4, -2.0, -4], [4, -2.0, -4], [-4, -2.0, 6]),    # ceiling
        plane([-2.4, -2.0, -4], [-2.4, 2.6, -4], [-2.4, -2.0, 6]),  # left wall
        plane([3.5, -2.0, -4], [3.5, 2.6, -4], [3.5, -2.0, 6]),     # right wall
    ]


ROOM_WALLS = None


def _make_texture(w: int, h: int, seed: int, cell: int = 56) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    base = rng.integers(60, 220, (max(h // 6, 2), max(w // 6, 2), 3), dtype=np.uint8)
    tex = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    for x in range(0, w, cell):
        cv2.line(tex, (x, 0), (x, h - 1), (25, 25, 25), 3)
    for y in range(0, h, cell):
        cv2.line(tex, (0, y), (w - 1, y), (25, 25, 25), 3)
    for _ in range(16):
        cx, cy = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        r = int(rng.integers(12, 36))
        color = tuple(int(v) for v in rng.integers(30, 230, 3))
        cv2.circle(tex, (cx, cy), r, color, -1)
        cv2.circle(tex, (cx, cy), r, (18, 18, 18), 2)
    return tex


def get_ray_map(cam: KannalaBrandt4, cache: dict | None = None):
    """Cached [H,W,3] float32 unit rays per pixel (per CameraModel instance)."""
    from ..geometry.ray_map import generate_ray_map

    if cache is None:
        rays, _ = generate_ray_map(cam)
        return rays.astype(np.float32)
    key = f"rays_{id(cam)}"
    if key not in cache:
        rays, _ = generate_ray_map(cam)
        cache[key] = rays.astype(np.float32)
    return cache[key]


def render_textured_room(
    cam: KannalaBrandt4,
    T_camera_room: SE3,
    ray_cache: dict | None = None,
    background: int = 15,
    texture_size: int = 1024,
) -> np.ndarray:
    """Exact software renderer: pixel -> ray -> plane intersection -> texture."""

    H_, W_ = cam.height, cam.width
    img = np.full((H_, W_, 3), background, dtype=np.uint8)
    rays = get_ray_map(cam, ray_cache)

    walls = wall_planes()
    order = sorted(range(len(walls)), key=lambda i: -walls[i]["depth"])  # far first
    for wi in order:
        w = walls[wi]
        origin_c = T_camera_room.apply(w["origin"])
        es_c = T_camera_room.apply_direction(w["e_s"])
        et_c = T_camera_room.apply_direction(w["e_t"])
        ls, lt = w["len_s"], w["len_t"]
        n_cam = np.cross(es_c, et_c)
        n_len = np.linalg.norm(n_cam)
        if n_len < 1e-12:
            continue
        n_cam = n_cam / n_len
        d_plane = float(n_cam @ origin_c)

        denom = rays @ n_cam
        valid = np.abs(denom) > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hit = np.where(valid, d_plane / np.where(valid, denom, 1.0), np.inf)
        valid &= (t_hit > 0.05) & (t_hit < 60.0)
        hit = rays * t_hit[..., None]

        rel = hit - origin_c
        s = ((rel @ es_c) / ls**2)
        t = ((rel @ et_c) / lt**2)
        inside = valid & (s >= 0) & (s < 1) & (t >= 0) & (t < 1)

        if not inside.any():
            continue
        tex = _make_texture(texture_size, texture_size, seed=200 + wi).astype(np.float32)
        xf = np.clip(s * (texture_size - 1), 0, texture_size - 1)
        yf = np.clip(t * (texture_size - 1), 0, texture_size - 1)
        x0 = np.floor(xf).astype(int)
        y0 = np.floor(yf).astype(int)
        x1 = np.minimum(x0 + 1, texture_size - 1)
        y1 = np.minimum(y0 + 1, texture_size - 1)
        wx = (xf - x0)[inside][:, None]
        wy = (yf - y0)[inside][:, None]
        c00 = tex[y0[inside], x0[inside]]
        c01 = tex[y0[inside], x1[inside]]
        c10 = tex[y1[inside], x0[inside]]
        c11 = tex[y1[inside], x1[inside]]
        img[inside] = np.clip(
            c00 * (1 - wx) * (1 - wy) + c01 * wx * (1 - wy)
            + c10 * (1 - wx) * wy + c11 * wx * wy, 0, 255
        ).astype(np.uint8)
    return img


def add_motion_blur_free_shift(img: np.ndarray) -> np.ndarray:  # pragma: no cover
    return img


def render_textured_room_ss(
    cam: KannalaBrandt4,
    T_camera_room: SE3,
    ray_cache: dict | None = None,
    ss: int = 2,
    **kwargs,
) -> np.ndarray:
    """Anti-aliased variant: render at ss× resolution with the scaled model,
    then downsample. Removes rasterization bias for sub-pixel-accurate QA."""
    import cv2

    cam2 = KannalaBrandt4(
        cam.fx * ss, cam.fy * ss,
        (cam.cx + 0.5) * ss - 0.5, (cam.cy + 0.5) * ss - 0.5,
        cam.k[0], cam.k[1], cam.k[2], cam.k[3],
        cam.width * ss, cam.height * ss,
    )
    big = render_textured_room(cam2, T_camera_room, ray_cache=ray_cache, **kwargs)
    return cv2.resize(big, (cam.width, cam.height), interpolation=cv2.INTER_AREA)

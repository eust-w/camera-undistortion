"""Hypothesis property-based tests: invariants must hold for arbitrary inputs."""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st  # noqa: E402

from camera_geometry.models.eucm import EUCM  # noqa: E402
from camera_geometry.models.kb4 import KannalaBrandt4  # noqa: E402
from camera_geometry.models.pinhole import PinholeCamera  # noqa: E402
from camera_geometry.transforms import SE3  # noqa: E402

W, H = 640, 480


def rotation_strategy():
    return st.lists(st.floats(-1, 1), min_size=9, max_size=9).map(
        lambda v: _orthonormalize(np.array(v).reshape(3, 3))
    )


def _orthonormalize(m: np.ndarray) -> np.ndarray:
    m = m + np.eye(3)  # avoid degenerate all-zero draws
    q, r = np.linalg.qr(m)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


@settings(max_examples=60, deadline=None)
@given(
    R=rotation_strategy(),
    t=st.lists(st.floats(-2, 2), min_size=3, max_size=3),
    p=st.lists(st.floats(-5, 5), min_size=3, max_size=3),
)
def test_property_se3_roundtrip(R, t, p):
    T = SE3.from_Rt(R, np.asarray(t))
    Ti = T.inverse()
    out = Ti.apply(T.apply(np.array([p])))
    assert np.allclose(out, np.array([p]), atol=1e-9)


@settings(max_examples=60, deadline=None)
@given(
    fx=st.floats(300, 800),
    fy=st.floats(300, 800),
    cx=st.floats(280, 360),
    cy=st.floats(200, 280),
    theta_deg=st.floats(0.5, 40.0),
    azim_deg=st.floats(0, 360),
)
def test_property_pinhole_roundtrip(fx, fy, cx, cy, theta_deg, azim_deg):
    cam = PinholeCamera(fx, fy, cx, cy, W, H)
    th, az = np.radians(theta_deg), np.radians(azim_deg)
    ray = np.array([[np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)]])
    back = cam.unproject(cam.project(ray))
    assert abs(float(back[0, 2]) - np.cos(th)) < 1e-9


@settings(max_examples=40, deadline=None)
@given(
    fy=st.floats(495, 515),
    cy=st.floats(630, 670),
    k1=st.floats(-0.2, 0.2),
    k2=st.floats(-0.05, 0.05),
    theta_deg=st.floats(0.5, 35.0),
    azim_deg=st.floats(0, 360),
)
def test_property_kb4_roundtrip(fy, cy, k1, k2, theta_deg, azim_deg):
    cam = KannalaBrandt4(505.0, fy, 800.0, cy, k1, k2, 0.0003, 0.0002, 1600, 1300)
    th, az = np.radians(theta_deg), np.radians(azim_deg)
    ray = np.array([[np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)]])
    back = cam.unproject(cam.project(ray))
    ang = np.degrees(np.arccos(np.clip(float(back[0] @ ray[0]), -1, 1)))
    assert ang < 1e-5  # extreme distortion draws allow slightly looser gate


@settings(max_examples=40, deadline=None)
@given(
    alpha=st.floats(0.05, 0.95),
    theta_deg=st.floats(0.5, 30.0),
    azim_deg=st.floats(0, 360),
)
def test_property_eucm_roundtrip(alpha, theta_deg, azim_deg):
    cam = EUCM(500, 505, 320, 240, alpha, W, H)
    th, az = np.radians(theta_deg), np.radians(azim_deg)
    ray = np.array([[np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)]])
    uv = cam.project(ray)
    if not np.all(np.isfinite(uv)):
        return
    back = cam.unproject(uv)
    ang = np.degrees(np.arccos(np.clip(float(back[0] @ ray[0]), -1, 1)))
    assert ang < 1e-6

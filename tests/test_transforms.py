import numpy as np
import pytest

from camera_geometry.models.base import to_numpy
from camera_geometry.transforms import SE3, rotx, roty, rotz


def random_se3(rng, angle=0.3, trans=0.5):
    R = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(R)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    Q = Q @ (roty(angle * rng.uniform()) @ rotx(angle * rng.uniform()))
    return SE3.from_Rt(Q, rng.uniform(-trans, trans, 3))


def test_identity():
    T = SE3.identity()
    p = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(T.apply(p), p)
    assert np.allclose(T.inverse().as_matrix(), np.eye(4))


def test_inverse_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = random_se3(rng)
        I = T.inverse() @ T
        assert np.allclose(I.R, np.eye(3), atol=1e-12)
        assert np.allclose(I.t, np.zeros(3), atol=1e-12)


def test_apply_direction_no_translation():
    T = random_se3(np.random.default_rng(1))
    d = np.array([[0.1, -0.2, 1.0]])
    out = T.apply(d) - T.apply_direction(d)
    assert np.allclose(out, np.tile(T.t, (1, 1)), atol=1e-12)


def test_point_transform_compose():
    rng = np.random.default_rng(2)
    T_ab = random_se3(rng)
    T_bc = random_se3(rng)
    T_ac = T_ab @ T_bc
    p_b = rng.uniform(-1, 1, 3)
    expected = T_ab.apply(T_bc.apply(p_b[None])[0])
    assert np.allclose(T_ac.apply(p_b[None])[0], expected, atol=1e-12)


@pytest.mark.parametrize("rot", [rotx(0.3), roty(-0.7), rotz(2.1)])
def test_known_rotations(rot):
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-14)
    assert abs(np.linalg.det(rot) - 1) < 1e-14


def test_invalid_rotation_rejected():
    with pytest.raises(ValueError):
        SE3.from_Rt(np.zeros((3, 3)), [0, 0, 0])


def test_camera_center_convention():
    R = roty(0.2)
    t = np.array([0.12086, 0.0, 0.0])
    T_left_right = SE3.from_Rt(R, t)
    # P_L = R @ P_R + t  =>  right center in left frame == t
    from camera_geometry.transforms import camera_center_in_reference

    assert np.allclose(camera_center_in_reference(T_left_right), [0.12086, 0.0, 0.0])
    # and transforming the right-frame origin must land exactly there
    assert np.allclose(T_left_right.apply(np.zeros((1, 3)))[0], [0.12086, 0.0, 0.0])


def test_to_numpy_accepts_lists():
    arr = to_numpy([[1, 2], [3, 4]])
    assert isinstance(arr, np.ndarray)

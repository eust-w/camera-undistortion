import numpy as np

from camera_geometry.models.kb4 import KannalaBrandt4
from camera_geometry.geometry.ray_map import generate_ray_map, load_ray_map, save_ray_map
from camera_geometry.geometry.triangulation import triangulate_midpoint
from camera_geometry.transforms import SE3


def make_kb4():
    return KannalaBrandt4(504.97497043, 504.85320543, 825.88761829, 644.70496968,
                          0.07305208, -0.01165996, 0.00033907, 0.00027176, 160, 130)


def test_ray_map_shape_and_consistency():
    cam = make_kb4()
    rays, valid = generate_ray_map(cam)
    assert rays.shape == (130, 160, 3)
    assert valid.shape == (130, 160)
    uv = np.array([[80.0, 65.0], [10.0, 10.0]])
    manual = np.asarray(cam.unproject(uv))
    for (u, v), m in zip(uv, manual):
        if valid[int(v), int(u)]:
            assert np.allclose(rays[int(v), int(u)], m, atol=1e-5)


def test_ray_map_save_load_fp16(tmp_path):
    cam = make_kb4()
    rays, valid = generate_ray_map(cam)
    p = tmp_path / "rm.npz"
    save_ray_map(p, rays, valid, fp16=True)
    rays2, valid2 = load_ray_map(p)
    assert rays2.dtype == np.float32
    assert np.allclose(rays2.astype(np.float16).astype(np.float64), rays, atol=1e-3)
    assert (valid == valid2).all()


def test_triangulation_synthetic():
    T_left_right = SE3.from_Rt(np.eye(3), [0.12, 0, 0])
    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.uniform(-1, 1, 200),
        rng.uniform(-1, 1, 200),
        rng.uniform(1.0, 4.0, 200),
    ])
    d_l = pts / np.linalg.norm(pts, axis=-1, keepdims=True)
    pts_r = T_left_right.inverse().apply(pts)
    d_r = pts_r / np.linalg.norm(pts_r, axis=-1, keepdims=True)
    tri, res = triangulate_midpoint(d_l, d_r, T_left_right)
    err = np.linalg.norm(tri - pts, axis=-1)
    good = np.isfinite(res) & (res < 0.01)
    assert good.sum() > 150
    assert float(np.median(err[good])) < 1e-6

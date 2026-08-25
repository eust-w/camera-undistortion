import numpy as np

from camera_geometry.models.kb4 import KannalaBrandt4
from camera_geometry.models.pinhole import PinholeCamera
from camera_geometry.stereo.rectify import build_stereo_rectifier
from camera_geometry.transforms import SE3, roty
from camera_geometry.virtual_camera.presets import create_virtual_camera
from camera_geometry.virtual_camera.remap import create_remap_grid


def make_kb4(seed=0):
    rng = np.random.default_rng(seed)
    return KannalaBrandt4(
        126.25 + rng.uniform(-0.4, 0.4), 126.22 + rng.uniform(-0.4, 0.4),
        200.0 + rng.uniform(-2, 2), 162.0 + rng.uniform(-1, 1),
        0.072, -0.0117, 0.00034, 0.00027, 400, 325,
    )


def test_virtual_grid_identity_for_pinhole():
    src = PinholeCamera.from_fov(200, 200, 90.0)
    dst = PinholeCamera.from_fov(100, 100, 90.0)
    grid, valid = create_remap_grid(src, dst)
    assert grid.shape == (100, 100, 2)
    assert valid.all()
    # center of dst must map to center-ish region of src (same optical axis)
    c = grid[50, 50]
    assert abs(c[0] - (src.width - 1) / 2) < 2.0
    assert abs(c[1] - (src.height - 1) / 2) < 2.0


def test_virtual_grid_kb4_center_alignment():
    cam = make_kb4()
    vcam = create_virtual_camera("ego_100")
    grid, valid = create_remap_grid(cam, vcam)
    # grid stores integer pixels: take the integer center pixel and its own ray
    ui, vi = vcam.width // 2, vcam.height // 2
    ray_i = np.asarray(vcam.unproject(np.array([[float(ui), float(vi)]])))
    uv_c = np.asarray(cam.project(ray_i))[0]
    got = grid[vi, ui]
    assert np.allclose(got, uv_c, atol=1e-4)
    assert valid.mean() > 0.9


def _rig(verge_deg=3.0, baseline=0.12):
    cam_l, cam_r = make_kb4(1), make_kb4(2)
    R = roty(np.radians(verge_deg))
    return cam_l, cam_r, SE3.from_Rt(R, np.array([baseline, 0.0, 0.0]))


def test_rectifier_shared_intrinsics_and_horizontal_epipolar():
    cam_l, cam_r, T = _rig()
    rig = build_stereo_rectifier(cam_l, cam_r, T, 320, 260, fov_x_deg=100.0)

    assert abs(rig.baseline_m - 0.12) < 1e-12
    K = rig.virtual_camera.K
    assert np.allclose(K[1, 1], rig.virtual_camera.fy)
    assert np.isclose(rig.virtual_camera.cx, (rig.virtual_camera.width - 1) / 2)
    assert np.isclose(rig.virtual_camera.cy, (rig.virtual_camera.height - 1) / 2)

    # project a few world points through both physical cams -> rectified y equal
    rng = np.random.default_rng(5)
    pts_l = rng.normal(size=(300, 3)) * [1.5, 0.8, 1.0]
    pts_l[:, 2] += 2.0

    pts_r = T.inverse().apply(pts_l)
    d_l = np.asarray(cam_l.pixel_to_ray(np.asarray(cam_l.project(pts_l))))
    d_r = np.asarray(cam_r.pixel_to_ray(np.asarray(cam_r.project(pts_r))))

    d_v_from_l = (rig.rotation_virtual_from_left @ d_l.T).T
    d_v_from_r = (rig.rotation_virtual_from_right @ d_r.T).T
    uv_l = rig.virtual_camera.project(d_v_from_l)
    uv_r = rig.virtual_camera.project(d_v_from_r)
    dy = np.abs(uv_l[:, 1] - uv_r[:, 1])
    assert float(np.median(dy)) < 1e-6, f"epipolar vertical residual {np.median(dy)}"

    # disparity consistency: u_l - u_r == fx*B/Z
    depth = pts_l[:, 2]
    disp = uv_l[:, 0] - uv_r[:, 0]
    expected = rig.virtual_camera.fx * rig.baseline_m / depth
    ok = disp > 0
    assert np.allclose(disp[ok], expected[ok], rtol=1e-6)


def test_rectifier_grids_map_into_source_bounds():
    cam_l, cam_r, T = _rig()
    rig = build_stereo_rectifier(cam_l, cam_r, T, 320, 260, fov_x_deg=100.0)
    for grid in (rig.grid_left, rig.grid_right):
        assert np.all(np.isfinite(grid))
        assert grid[..., 0].max() < cam_l.width + 1
        assert grid[..., 0].min() > -1

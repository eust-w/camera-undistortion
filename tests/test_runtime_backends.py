import numpy as np
import pytest

from camera_geometry.models.pinhole import PinholeCamera
from camera_geometry.runtime.opencv_backend import remap_image
from camera_geometry.virtual_camera.remap import create_remap_grid, to_torch_grid


def test_cv2_remap_identity_when_same_camera():
    src = PinholeCamera.from_fov(64, 64, 90.0)
    grid, _ = create_remap_grid(src, src)
    img = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    out = remap_image(img, grid)
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 0.5
    # corners preserved exactly by backward identity map
    assert np.array_equal(out[32], img[32])


def test_torch_grid_matches_cv2():
    torch = pytest.importorskip("torch")
    from camera_geometry.runtime.torch_backend import make_sampling_grid, remap_batch

    src = PinholeCamera.from_fov(64, 64, 100.0)
    dst = PinholeCamera.from_fov(32, 32, 100.0)
    grid, _ = create_remap_grid(src, dst)

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (1, 3, 64, 64), dtype=np.uint8).astype(np.float32)
    out_cv = np.stack([remap_image(img[0].transpose(1, 2, 0).astype(np.uint8), grid) for _ in range(1)])

    sampling = make_sampling_grid(grid, src.width, src.height)
    out_t = remap_batch(torch.from_numpy(img), sampling)
    diff = np.abs(out_t.detach().numpy()[0].transpose(1, 2, 0) - out_cv[0]).mean()
    assert diff < 1.5


def test_to_torch_grid_range():
    grid = np.random.default_rng(1).uniform(0, 63, (16, 16, 2)).astype(np.float32)
    g = to_torch_grid(grid, 64, 64)
    assert g.shape == (16, 16, 2)
    assert -1.0001 <= g.min() and g.max() <= 1.0001

"""OpenCV runtime backend (PoC / CPU path)."""

from __future__ import annotations

import cv2
import numpy as np

from ..virtual_camera.remap import to_cv_maps


def remap_image(image: np.ndarray, grid: np.ndarray, border_value: int = 0) -> np.ndarray:
    mapx, mapy = to_cv_maps(grid)
    return cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def remap_images(images: list[np.ndarray], grid: np.ndarray) -> list[np.ndarray]:
    mapx, mapy = to_cv_maps(grid)
    return [
        cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        for img in images
    ]

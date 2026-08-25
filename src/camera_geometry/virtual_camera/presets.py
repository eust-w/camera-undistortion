"""Virtual pinhole view presets for VLM / VLA / plain CV consumers."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.pinhole import PinholeCamera


@dataclass(frozen=True)
class VirtualPreset:
    name: str
    width: int = 640
    height: int = 640
    fov_x_deg: float = 100.0

    def build(self) -> PinholeCamera:
        return PinholeCamera.from_fov(self.width, self.height, self.fov_x_deg)


EGO_PRESETS: dict[str, VirtualPreset] = {
    "ego_90": VirtualPreset("ego_90", 640, 640, 90.0),
    "ego_100": VirtualPreset("ego_100", 640, 640, 100.0),
    "ego_110": VirtualPreset("ego_110", 640, 640, 110.0),
}

DEFAULT_PRESET = "ego_100"


def create_virtual_camera(name: str | None = None, **overrides) -> PinholeCamera:
    if name is not None and overrides:
        raise ValueError("pass either preset name or explicit overrides, not both")
    if name is not None:
        return EGO_PRESETS[name].build()
    return VirtualPreset("custom", **overrides).build()

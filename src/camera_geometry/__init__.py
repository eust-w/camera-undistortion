"""Auto Camera Geometry.

Canonical representation: Raw image + CameraModel + Extrinsics + Timestamp.
Core abstraction: pixel <-> ray. Undistortion is only a Virtual Camera output.
"""

from .schema import CODE_VERSION, SCHEMA_VERSION, DeviceCalibration
from .transforms import SE3

__version__ = CODE_VERSION
__all__ = ["CODE_VERSION", "SCHEMA_VERSION", "DeviceCalibration", "SE3"]

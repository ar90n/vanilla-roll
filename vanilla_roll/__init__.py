try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__: str = version(__name__)
except PackageNotFoundError:
    __version__: str = "unknown"

from . import anatomy_orientation, camera, camera_sequence, io, rendering, volume
from .usecase import render, render_horizontal_rotations, render_vertical_rotations

__all__ = [
    "io",
    "render",
    "render_horizontal_rotations",
    "render_vertical_rotations",
    "anatomy_orientation",
    "camera",
    "rendering",
    "camera_sequence",
    "volume",
]

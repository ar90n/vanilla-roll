from dataclasses import dataclass
from typing import Callable, TypeAlias

import vanilla_roll.array_api as xp
from vanilla_roll.camera import Camera
from vanilla_roll.geometry.element import Orientation, Vector


@dataclass(frozen=True)
class RenderingResult:
    image: xp.Array
    spacing: Vector | None
    origin: Vector | None
    orientation: Orientation | None


Renderer: TypeAlias = Callable[[Camera, tuple[int, int]], RenderingResult]

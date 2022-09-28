from dataclasses import dataclass
from typing import Protocol, TypeAlias

import vanilla_roll.array_api as xp
from vanilla_roll.camera import Camera
from vanilla_roll.geometry.element import Orientation, Vector


@dataclass(frozen=True)
class ColorImage:
    r: xp.Array
    g: xp.Array
    b: xp.Array


@dataclass(frozen=True)
class MonoImage:
    l: xp.Array


Image: TypeAlias = ColorImage | MonoImage


@dataclass(frozen=True)
class RenderingResult:
    image: Image
    spacing: Vector | None
    origin: Vector | None
    orientation: Orientation | None


class Renderer(Protocol):
    def __call__(
        self,
        camera: Camera,
        spacing: float | None = ...,
    ) -> RenderingResult:
        ...

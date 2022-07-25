from dataclasses import dataclass
from functools import cached_property

import vanilla_roll.array_api as xp
from vanilla_roll.geometry.element import (
    Frame,
    Orientation,
    Vector,
    as_array,
)
from vanilla_roll.geometry.linalg import normalize_vector
from vanilla_roll.geometry.conversion import Conversion
from vanilla_roll.validation import (
    IsFinite,
    IsGreaterEqualThan,
    IsGreaterThan,
    Validator,
)


@dataclass(frozen=True)
class ViewVolume:
    width: float
    height: float
    far: float
    near: float

    def __post_init__(self) -> None:
        size_validator = Validator(rules=[IsGreaterThan(0.0), IsFinite()])
        for f in ("width", "height"):
            if exception := size_validator(f, getattr(self, f)):
                raise exception

        depth_validator = Validator(rules=[IsGreaterEqualThan(0.0), IsFinite()])
        for f in ("far", "near"):
            if exception := depth_validator(f, getattr(self, f)):
                raise exception

        if self.far < self.near:
            raise ValueError(
                f"far ({self.far}) must be greater than near ({self.near})"
            )


@dataclass(frozen=True)
class Camera:
    frame: Frame
    view_volume: ViewVolume

    @property
    def forward(self) -> Vector:
        return self.frame.orientation.k

    @cached_property
    def up(self) -> Vector:
        return Vector.of_array(-as_array(self.frame.orientation.j))

    @cached_property
    def screen_origin(self) -> Vector:
        dir_i = as_array(normalize_vector(self.screen_orientation.i))
        dir_j = as_array(normalize_vector(self.screen_orientation.j))

        return Vector.of_array(
            as_array(self.screen_center)
            - (self.view_volume.width / 2.0) * dir_i
            - (self.view_volume.height / 2.0) * dir_j
        )

    @cached_property
    def screen_center(self) -> Vector:
        center = (
            as_array(self.frame.origin)
            # + self.view_volume.far * self.forward.to_array()
            # + as_array(self.forward)
        )
        return Vector.of_array(center)

    @property
    def screen_shape(self) -> tuple[int, int]:
        return (int(self.view_volume.height), int(self.view_volume.width))

    @property
    def screen_orientation(self) -> Orientation:
        return self.frame.orientation

    @cached_property
    def screen_frame(self) -> Frame:
        return Frame(
            origin=self.screen_origin,
            orientation=self.screen_orientation,
        )

    @cached_property
    def view_matrix(self) -> xp.Array:
        return xp.linalg.inv(as_array(self.frame))

    def apply(self, convert: Conversion) -> "Camera":
        return Camera(
            frame=convert(self.frame),
            view_volume=self.view_volume,
        )

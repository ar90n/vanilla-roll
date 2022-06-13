from dataclasses import dataclass
from functools import cached_property

import vanilla_roll.array_api as xp
from vanilla_roll.geometry import Direction, Orientation, Point
from vanilla_roll.validation import (
    IsFinite,
    IsGreaterEqualThan,
    IsGreaterThan,
    Validator,
)


@dataclass
class Screen:
    width: float
    height: float
    distance: float

    def __post_init__(self) -> None:
        size_validator = Validator(rules=[IsGreaterThan(0.0), IsFinite()])
        if exception := size_validator("width", self.width):
            raise exception
        if exception := size_validator("height", self.height):
            raise exception

        distance_validator = Validator(rules=[IsGreaterEqualThan(0.0), IsFinite()])
        if exception := distance_validator("distance", self.distance):
            raise exception


@dataclass(frozen=True)
class Camera:
    position: Point
    forward: Direction
    up: Direction
    screen: Screen

    def __post_init__(self) -> None:
        projected_up = (
            self.up.to_array()
            - (self.up.to_array() @ self.forward.to_array()) * self.forward.to_array()
        )
        if xp.all(xp.abs(projected_up) < 1e-12):
            raise ValueError("forward and up are parallel")

        object.__setattr__(self, "up", Direction.of_array(projected_up))

    @cached_property
    def screen_origin(self) -> Point:
        return Point.of_array(
            self.screen_center.to_array()
            - (self.screen.width / 2.0) * self.screen_orientation.i.to_array()
            - (self.screen.height / 2.0) * self.screen_orientation.j.to_array()
        )

    @cached_property
    def screen_center(self) -> Point:
        center = (
            self.position.to_array() + self.screen.distance * self.forward.to_array()
        )
        return Point.of_array(center)

    @cached_property
    def screen_orientation(self) -> Orientation:
        column_direction = Direction.of_array(-self.up.to_array())
        row_direction = Direction.of_array(
            xp.linalg.cross(self.forward.to_array(), column_direction.to_array())
        )

        return Orientation(
            i=row_direction,
            j=column_direction,
            k=self.forward,
        )

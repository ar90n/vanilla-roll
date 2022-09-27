from dataclasses import dataclass
from functools import cached_property
from typing import NoReturn

import vanilla_roll.array_api as xp
from vanilla_roll.anatomy_orientation import AnatomyOrientation
from vanilla_roll.geometry.element import Frame, Vector
from vanilla_roll.geometry.linalg import norm


@dataclass(frozen=True)
class Volume:

    data: xp.Array
    frame: Frame
    anatomy_orientation: AnatomyOrientation | None = None

    def __post_init__(self) -> None | NoReturn:
        if self.data.ndim != 3:
            raise ValueError(f"Expected 2D array, got {self.data.ndim}D array")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @cached_property
    def spacing(self) -> Vector:
        return Vector(
            k=norm(self.frame.orientation.k),
            j=norm(self.frame.orientation.j),
            i=norm(self.frame.orientation.i),
        )

    @cached_property
    def diagonal_length(self) -> float:
        return (
            (self.spacing.k * self.shape[0]) ** 2
            + (self.spacing.j * self.shape[1]) ** 2
            + (self.spacing.i * self.shape[2]) ** 2
        ) ** 0.5

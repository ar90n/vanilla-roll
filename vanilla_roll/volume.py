from dataclasses import dataclass
from typing import NoReturn

import vanilla_roll.array_api as xp
from vanilla_roll.geometry import CoordinateSystem


@dataclass(frozen=True)
class Volume:

    data: xp.Array
    coordinate_system: CoordinateSystem

    def __post_init__(self) -> None | NoReturn:
        if self.data.ndim != 3:
            raise ValueError(f"Expected 2D array, got {self.data.ndim}D array")

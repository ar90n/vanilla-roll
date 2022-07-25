from dataclasses import dataclass
from typing import NoReturn

import vanilla_roll.array_api as xp
from vanilla_roll.geometry.element import Frame


@dataclass(frozen=True)
class Volume:

    data: xp.Array
    frame: Frame

    def __post_init__(self) -> None | NoReturn:
        if self.data.ndim != 3:
            raise ValueError(f"Expected 2D array, got {self.data.ndim}D array")

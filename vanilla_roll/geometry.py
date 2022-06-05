from dataclasses import InitVar, dataclass, fields
from typing import NoReturn

from vanilla_roll.validation import IsFinite, IsGreaterThan, Validator


@dataclass(frozen=True)
class Spacing2d:
    """Spacing2d is a 2D spacing in mm.

    >>> spacing = Spacing2d(column=1.0, row=2.0)
    >>> (spacing.column, spacing.row)
    (1.0, 2.0)
    """

    column: float
    row: float
    validator: InitVar[Validator] = Validator(rules=[IsGreaterThan(0.0), IsFinite()])

    def __post_init__(self, validator: Validator) -> None | NoReturn:
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception


@dataclass(frozen=True)
class Spacing:
    """Space is a 3D spacing in mm."""

    layer: float
    column: float
    row: float
    validator: InitVar[Validator] = Validator(rules=[IsGreaterThan(0.0), IsFinite()])

    def __post_init__(self, validator: Validator) -> None | NoReturn:
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception


@dataclass(frozen=True)
class Origin:
    """Origin is a 3D origin.

    >>> o = Origin(x=1.0, y=2.0, z=3.0)
    >>> (o.x, o.y, o.z)
    (1.0, 2.0, 3.0)
    """

    x: float
    y: float
    z: float
    validator: InitVar[Validator] = Validator(rules=[IsFinite()])

    def __post_init__(self, validator: Validator) -> None | NoReturn:
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception


@dataclass(frozen=True)
class Direction:
    """Direction is a 3D direction.

    >>> d = Direction(1.0, 2.0, 3.0)
    >>> (d.x, d.y, d.z)
    (0.2672612419124244, 0.5345224838248488, 0.8017837257372732)
    """

    x: float
    y: float
    z: float
    validator: InitVar[Validator] = Validator(rules=[IsFinite()])

    def __post_init__(self, validator: Validator) -> None | NoReturn:
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception

        norm = (self.x**2 + self.y**2 + self.z**2) ** 0.5
        if norm == 0.0:
            raise ValueError("Direction vector cannot be zero.")
        object.__setattr__(self, "x", self.x / norm)
        object.__setattr__(self, "y", self.y / norm)
        object.__setattr__(self, "z", self.z / norm)


@dataclass(frozen=True)
class Orientation:
    """Orientation is an orientation in 3D space by directions of each axes.

    >>> d = Orientation(layer=Direction(1.0, 0.0, 0.0), column=Direction(0.0, 1.0, 0.0), row=Direction(0.0, 0.0, 1.0))
    >>> d.layer
    Direction(x=1.0, y=0.0, z=0.0)
    >>> d.column
    Direction(x=0.0, y=1.0, z=0.0)
    >>> d.row
    Direction(x=0.0, y=0.0, z=1.0)
    """

    layer: Direction
    column: Direction
    row: Direction


@dataclass(frozen=True)
class Coordinate2d:
    """Coordinate2d is a 2D coordinate in slice space.

    >>> c = Coordinate2d(column=1.0, row=2.0)
    >>> (c.column, c.row)
    (1.0, 2.0)
    """

    column: float
    row: float
    validator: InitVar[Validator] = Validator(rules=[IsFinite()])

    def __post_init__(self, validator: Validator) -> None | NoReturn:
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception


@dataclass(frozen=True)
class Coordinate:
    """Coordinate is a 3D coordinate in volume space.

    >>> c = Coordinate(layer=1.0, column=2.0, row=3.0)
    >>> (c.layer, c.column, c.row)
    (1.0, 2.0, 3.0)
    """

    layer: float
    column: float
    row: float
    validator: InitVar[Validator] = Validator(rules=[IsFinite()])

    def __post_init__(self, validator: Validator) -> None | NoReturn:
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception

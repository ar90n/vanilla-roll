import math
from dataclasses import dataclass, fields
from typing import Any, NoReturn, overload

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
from vanilla_roll.validation import IsFinite, IsGreaterThan, Validator


@dataclass(frozen=True)
class Spacing:
    """Space is a 3D spacing in mm."""

    i: float
    j: float
    k: float

    def __post_init__(self) -> None | NoReturn:
        validator = Validator(rules=[IsGreaterThan(0.0), IsFinite()])
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception

    @classmethod
    def of_array(cls, array: xp.Array) -> "Spacing":
        if (elms := math.prod(array.shape)) != 3:
            raise ValueError(f"Expected 3 elements in array, got {elms}")
        if array.shape != (3,):
            array = xp.reshape(array, (-1,))

        k, j, i = (float(array[ai]) for ai in range(3))
        return cls(i=i, j=j, k=k)

    def to_array(self) -> xp.Array:
        return xp.asarray([self.k, self.j, self.i])

    def to_homogeneous(self) -> xp.Array:
        return xp.asarray([self.k, self.j, self.i, 1.0])


@dataclass(frozen=True)
class Vector:
    """Vector is a 3D Vector.

    >>> d = Vector(1.0, 2.0, 3.0)
    >>> (d.x, d.y, d.z)
    (1.0, 2.0, 3.0)
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None | NoReturn:
        validator = Validator(rules=[IsFinite()])
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception

    @classmethod
    def of_array(cls, array: xp.Array) -> "Vector":
        if (elms := math.prod(array.shape)) != 3:
            raise ValueError(f"Expected 3 elements in array, got {elms}")
        if array.shape != (3,):
            array = xp.reshape(array, (-1,))

        z, y, x = (float(array[ai]) for ai in range(3))
        return cls(x=x, y=y, z=z)

    def to_array(self) -> xp.Array:
        return xp.asarray([self.z, self.y, self.x])

    def to_homogeneous(self) -> xp.Array:
        return xp.asarray([self.z, self.y, self.x, 1.0])


@dataclass(frozen=True)
class Direction(Vector):
    @classmethod
    def of(cls, vector: Vector) -> "Direction":
        return cls(vector.x, vector.y, vector.z)

    def __post_init__(self) -> None | NoReturn:
        super().__post_init__()
        norm = (self.x**2 + self.y**2 + self.z**2) ** 0.5
        if norm == 0.0:
            raise ValueError("Direction vector cannot be zero.")
        object.__setattr__(self, "x", self.x / norm)
        object.__setattr__(self, "y", self.y / norm)
        object.__setattr__(self, "z", self.z / norm)

    @classmethod
    def of_array(cls, array: xp.Array) -> "Direction":
        return cls.of(super().of_array(array))


@dataclass(frozen=True)
class Point:
    """Point is a 3D coordinate.

    >>> c = Point(1.0, 2.0, 3.0)
    >>> (c.x, c.y, c.z)
    (1.0, 2.0, 3.0)
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None | NoReturn:
        validator = Validator(rules=[IsFinite()])
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception

    @classmethod
    def of_array(cls, array: xp.Array) -> "Point":
        if (elms := math.prod(array.shape)) != 3:
            raise ValueError(f"Expected 3 elements in array, got {elms}")
        if array.shape != (3,):
            array = xp.reshape(array, (-1,))

        z, y, x = (float(array[ai]) for ai in range(3))
        return cls(x=x, y=y, z=z)

    def to_array(self) -> xp.Array:
        return xp.asarray([self.z, self.y, self.x])

    def to_homogeneous(self) -> xp.Array:
        return xp.asarray([self.z, self.y, self.x, 1.0])


Primitive = Point | Vector | Direction | Spacing


@dataclass(frozen=True)
class Orientation:
    """Orientation is an orientation in 3D space by directions of each axes.

    >>> d = Orientation(Direction(1.0, 0.0, 0.0), Direction(0.0, 1.0, 0.0), Direction(0.0, 0.0, 1.0))
    >>> d.i
    Direction(x=1.0, y=0.0, z=0.0)
    >>> d.j
    Direction(x=0.0, y=1.0, z=0.0)
    >>> d.k
    Direction(x=0.0, y=0.0, z=1.0)
    """

    i: Direction
    j: Direction
    k: Direction

    def to_array(self) -> xp.Array:
        k = xp.reshape(self.k.to_array(), (1, 3))
        j = xp.reshape(self.j.to_array(), (1, 3))
        i = xp.reshape(self.i.to_array(), (1, 3))
        return xp.concat([k, j, i], axis=0)

    def to_homogeneous(self) -> xp.Array:
        r = xpe.diag(xp.ones(4))
        r[:3, :3] = self.to_array()
        return r


@dataclass(frozen=True)
class CoordinateSystem:
    """CoordinateSystem is a coordinate system in 3D space."""

    origin: Point
    orientation: Orientation
    spacing: Spacing

    def to_array(self) -> xp.Array:
        s = xpe.diag(self.spacing.to_homogeneous())
        r = self.orientation.to_homogeneous()

        t = xpe.diag(xp.ones(4))
        t[:, 3] = self.origin.to_homogeneous()
        return t @ r @ s


@dataclass(frozen=True)
class Transform:
    src: xp.Array
    dst: xp.Array

    @classmethod
    def of(cls, /, src: CoordinateSystem, dst: CoordinateSystem) -> "Transform":
        return cls(
            xp.astype(src.to_array(), xp.float64), xp.astype(dst.to_array(), xp.float64)
        )

    @overload
    def apply(self, target: Point) -> Point:
        ...

    @overload
    def apply(self, target: Vector) -> Vector:
        ...

    @overload
    def apply(self, /, target: xp.Array) -> xp.Array:
        ...

    def apply(self, target: Any) -> Any:
        if isinstance(target, Point):
            dst_point = xp.linalg.solve(self.dst, self.src @ target.to_homogeneous())
            return Point(z=dst_point[0], y=dst_point[1], x=dst_point[2])  # type: ignore
        elif isinstance(target, Vector):
            dst_vector = xp.linalg.solve(self.dst, self.src @ target.to_homogeneous())
            return Vector(z=dst_vector[0], y=dst_vector[1], x=dst_vector[2])  # type: ignore
        elif isinstance(target, xp.Array):
            org_shape = target.shape
            if target.shape == (3,):
                target = xp.reshape(target, (3, 1))
            if target.ndim != 2 or target.shape[0] != 3:
                raise ValueError("Target array must be a 3xN matrix.")
            if target.dtype != xp.float64:
                target = xp.astype(target, xp.float64)

            homogeneous_target = xp.concat(
                [
                    target,
                    xp.reshape(xp.ones(target.shape[1], dtype=xp.float64), (1, -1)),
                ],
                axis=0,
            )
            result = xp.linalg.solve(self.dst, self.src @ homogeneous_target)[:3, :]
            return xp.reshape(result, org_shape)
        raise TypeError("Transform.apply() only accepts Point or xp.Array.")


world_origin = Point(0.0, 0.0, 0.0)
world_orientation = Orientation(
    Direction(1.0, 0.0, 0.0), Direction(0.0, 1.0, 0.0), Direction(0.0, 0.0, 1.0)
)
world_spacing = Spacing(1.0, 1.0, 1.0)
world_coordinate_system = CoordinateSystem(
    origin=world_origin,
    orientation=world_orientation,
    spacing=world_spacing,
)

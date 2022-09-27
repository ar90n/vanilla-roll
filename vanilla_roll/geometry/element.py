import math
from dataclasses import dataclass, fields
from typing import NoReturn

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
from vanilla_roll.validation import IsFinite, Validator


@dataclass(frozen=True)
class Vector:
    """Vector is a 3D Vector.

    >>> d = Vector(1.0, 2.0, 3.0)
    >>> (d.i, d.j, d.k)
    (1.0, 2.0, 3.0)
    """

    i: float
    j: float
    k: float

    def __post_init__(self) -> None | NoReturn:
        validator = Validator(rules=[IsFinite()])
        for f in fields(self):
            if exception := validator(f.name, getattr(self, f.name)):
                raise exception

    @classmethod
    def of_array(cls, array: xp.Array) -> "Vector":
        """Convert from array.

        >>> Vector.of_array(xp.asarray([1.0, 2.0, 3.0]))
        Vector(i=3.0, j=2.0, k=1.0)
        """

        if (elms := math.prod(array.shape)) != 3:
            raise ValueError(f"Expected 3 elements in array, got {elms}")
        if array.shape != (3,):
            array = xp.reshape(array, (-1,))

        k, j, i = (float(array[ai]) for ai in range(3))
        return cls(i=i, j=j, k=k)

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(
            i=self.i + other.i,
            j=self.j + other.j,
            k=self.k + other.k,
        )

    def __sub__(self, other: "Vector") -> "Vector":
        return self + (-1.0 * other)

    def __mul__(self, other: float) -> "Vector":
        return Vector(
            i=self.i * other,
            j=self.j * other,
            k=self.k * other,
        )

    def __rmul__(self, other: float) -> "Vector":
        return self * other

    def __matmul__(self, other: "Vector") -> float:
        return self.i * other.i + self.j * other.j + self.k * other.k


@dataclass(frozen=True)
class Orientation:
    """Orientation is an orientation in 3D space by vectors of each axes.

    >>> d = Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0))
    >>> d.i
    Vector(i=1.0, j=0.0, k=0.0)
    >>> d.j
    Vector(i=0.0, j=1.0, k=0.0)
    >>> d.k
    Vector(i=0.0, j=0.0, k=1.0)
    """

    i: Vector
    j: Vector
    k: Vector


@dataclass(frozen=True)
class Frame:
    """Frame is a frame in 3D space."""

    origin: Vector
    orientation: Orientation


def _as_array_from_vector(vector: Vector) -> xp.Array:
    return xp.asarray([vector.k, vector.j, vector.i], dtype=xp.float32)


def _as_array_from_orientation(orientation: Orientation) -> xp.Array:
    k = xp.reshape(_as_array_from_vector(orientation.k), (1, 3))
    j = xp.reshape(_as_array_from_vector(orientation.j), (1, 3))
    i = xp.reshape(_as_array_from_vector(orientation.i), (1, 3))
    return xp.concat([k, j, i], axis=0).T


def _as_array_from_frame(frame: Frame) -> xp.Array:
    a = xpe.diag(xp.ones(4, dtype=xp.float32))
    a[:3, :3] = _as_array_from_orientation(frame.orientation)
    a[:3, 3] = _as_array_from_vector(frame.origin)
    return a


def as_array(data: Vector | Orientation | Frame) -> xp.Array:
    """Convert to xp.Array.

    >>> origin = Vector(1.0, 2.0, 3.0)
    >>> orientation = Orientation(i=Vector(1.0, 0.0, 0.0), j=Vector(0.0, 1.0, 0.0), k=Vector(0.0, 0.0, 1.0))
    >>> as_array(origin)
    Array([3., 2., 1.], dtype=float32)
    >>> as_array(orientation)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)
    >>> _as_array_from_frame(Frame(origin, orientation))
    Array([[1., 0., 0., 3.],
           [0., 1., 0., 2.],
           [0., 0., 1., 1.],
           [0., 0., 0., 1.]], dtype=float32)
    """
    match data:
        case Vector():
            return _as_array_from_vector(data)
        case Orientation():
            return _as_array_from_orientation(data)
        case Frame():
            return _as_array_from_frame(data)
    raise ValueError(f"Unknown data type: {type(data)}")


def to_homogeneous(array: xp.Array) -> xp.Array:
    """Convert to homogeneous coordinates.

    >>> to_homogeneous(xp.asarray([1.0, 2.0, 3.0]))
    Array([1., 2., 3., 1.], dtype=float64)
    >>> to_homogeneous(xp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    Array([[1., 2.],
           [3., 4.],
           [5., 6.],
           [1., 1.]], dtype=float64)
    >>> to_homogeneous(xp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    Array([[1., 2., 3., 1.],
           [4., 5., 6., 1.]], dtype=float64)
    """

    if array.ndim == 1:
        return xp.concat([array, xp.asarray([1.0], dtype=array.dtype)])
    elif array.ndim == 2 and array.shape[0] == 3:
        return xp.concat(
            [array, xp.ones((1, array.shape[1]), dtype=array.dtype)], axis=0
        )
    elif array.ndim == 2 and array.shape[1] == 3:
        return xp.concat(
            [array, xp.ones((array.shape[0], 1), dtype=array.dtype)], axis=1
        )

    raise ValueError("Not implemented")


world_frame = Frame(
    origin=Vector(0.0, 0.0, 0.0),
    orientation=Orientation(
        Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)
    ),
)

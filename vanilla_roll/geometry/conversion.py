from typing import Protocol, overload

import vanilla_roll.array_api as xp
from vanilla_roll.geometry.element import (
    Frame,
    Orientation,
    Vector,
    as_array,
    to_homogeneous,
)


class Conversion(Protocol):
    @overload
    def __call__(self, target: Vector) -> Vector:
        ...

    @overload
    def __call__(self, target: Orientation) -> Orientation:
        ...

    @overload
    def __call__(self, target: Frame) -> Frame:
        ...

    @overload
    def __call__(self, target: xp.Array) -> xp.Array:
        ...

    def __call__(
        self, target: Vector | Orientation | Frame | xp.Array
    ) -> Vector | Orientation | Frame | xp.Array:
        ...


def _transform_vector(
    target: Vector, src_frame_mat: xp.Array, dst_frame_mat: xp.Array
) -> Vector:
    homogeneous_target = to_homogeneous(as_array(target))
    return Vector.of_array(
        xp.linalg.solve(dst_frame_mat, src_frame_mat @ homogeneous_target)[:3]
    )


def _transform_orientation(
    target: Orientation, src_frame_mat: xp.Array, dst_frame_mat: xp.Array
) -> Orientation:
    src_rot_mat = xp.eye(4, dtype=src_frame_mat.dtype)
    src_rot_mat[:3, :3] = src_frame_mat[:3, :3]
    dst_rot_mat = xp.eye(4, dtype=dst_frame_mat.dtype)
    dst_rot_mat[:3, :3] = dst_frame_mat[:3, :3]
    return Orientation(
        i=_transform_vector(target.i, src_rot_mat, dst_rot_mat),
        j=_transform_vector(target.j, src_rot_mat, dst_rot_mat),
        k=_transform_vector(target.k, src_rot_mat, dst_rot_mat),
    )


def _transform_frame(
    target: Frame, src_frame_mat: xp.Array, dst_frame_mat: xp.Array
) -> Frame:
    return Frame(
        _transform_vector(target.origin, src_frame_mat, dst_frame_mat),
        _transform_orientation(target.orientation, src_frame_mat, dst_frame_mat),
    )


def _transoform_array(
    target: xp.Array, src_frame_mat: xp.Array, dst_frame_mat: xp.Array
) -> xp.Array:
    org_shape = target.shape
    if target.shape == (3,):
        target = xp.reshape(target, (3, 1))
    if target.ndim != 2 or target.shape[0] != 3:
        raise ValueError("Target array must be a 3xN matrix.")
    if src_frame_mat.dtype != target.dtype:
        src_frame_mat = xp.astype(src_frame_mat, target.dtype)
    if dst_frame_mat.dtype != target.dtype:
        dst_frame_mat = xp.astype(dst_frame_mat, target.dtype)

    homogeneous_target = to_homogeneous(target)
    result = xp.linalg.solve(dst_frame_mat, src_frame_mat @ homogeneous_target)[:3, :]

    return xp.reshape(result, org_shape)


class Transformation(Conversion):
    """Create a transformation from src frame to dst frame.

    >>> src = Frame(\
            Vector(1.0, 2.0, 3.0),\
            Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0))\
        )
    >>> dst = Frame(\
            Vector(2.0, 1.0, 0.5),\
            Orientation(Vector(0.5, 0.5, 0.0), Vector(-0.5, 0.5, 0.0), Vector(0.0, 0.0, 1.0))\
        )
    >>> transform = Transformation(src, dst)
    >>> transform(Vector(1.0, 2.0, 3.0))
    Vector(i=3.0, j=3.0, k=5.5)
    >>> transform(Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, -0.5, 0.5), Vector(0.0, 0.5, 0.5)))
    Orientation(i=Vector(i=1.0, j=-1.0, k=0.0), j=Vector(i=-0.5, j=-0.5, k=0.5), k=Vector(i=0.5, j=0.5, k=0.5))
    >>> transform(Frame(\
            Vector(1.0, 2.0, 3.0),\
            Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0))\
        ))
    Frame(origin=Vector(i=3.0, j=3.0, k=5.5), orientation=Orientation(i=Vector(i=1.0, j=-1.0, k=0.0), j=Vector(i=1.0, j=1.0, k=0.0), k=Vector(i=0.0, j=0.0, k=1.0)))
    """  # noqa: E501

    _src_frame_mat: xp.Array
    _dst_frame_mat: xp.Array

    def __init__(self, src: Frame, dst: Frame) -> None:
        self._src_frame_mat = as_array(src)
        self._dst_frame_mat = as_array(dst)

    @overload
    def __call__(self, target: Vector) -> Vector:
        ...

    @overload
    def __call__(self, target: Orientation) -> Orientation:
        ...

    @overload
    def __call__(self, target: Frame) -> Frame:
        ...

    @overload
    def __call__(self, target: xp.Array) -> xp.Array:
        ...

    def __call__(self, target: Vector | Orientation | Frame | xp.Array):
        match target:
            case Vector():
                return _transform_vector(
                    target, self._src_frame_mat, self._dst_frame_mat
                )
            case Orientation():
                return _transform_orientation(
                    target, self._src_frame_mat, self._dst_frame_mat
                )
            case Frame():
                return _transform_frame(
                    target, self._src_frame_mat, self._dst_frame_mat
                )
            case xp.Array():
                return _transoform_array(
                    target, self._src_frame_mat, self._dst_frame_mat
                )
        raise ValueError(f"Unknown data type: {type(target)}")


def _permutate_vector(target: Vector, order: tuple[int, int, int]) -> Vector:
    k, j, i = ("kji"[i] for i in order)
    return Vector(
        i=getattr(target, i),
        j=getattr(target, j),
        k=getattr(target, k),
    )


def _permute_orientation(
    target: Orientation, order: tuple[int, int, int]
) -> Orientation:
    return Orientation(
        i=_permutate_vector(target.i, order),
        j=_permutate_vector(target.j, order),
        k=_permutate_vector(target.k, order),
    )


def _permutate_frame(target: Frame, order: tuple[int, int, int]) -> Frame:
    return Frame(
        origin=_permutate_vector(target.origin, order),
        orientation=_permute_orientation(target.orientation, order),
    )


def _permutate_array(target: xp.Array, order: tuple[int, int, int]) -> xp.Array:
    if target.shape == (3,):
        target = xp.reshape(target, (3, 1))
    if target.ndim != 2 or target.shape[0] != 3:
        raise ValueError("Target array must be a 3xN matrix.")
    return xp.stack([target[i, :] for i in order])


class Permutation(Conversion):
    """Create a permutation of given order.

    >>> perm = Permutation((1, 2, 0))
    >>> perm(Vector(1.0, 2.0, 3.0))
    Vector(i=3.0, j=1.0, k=2.0)
    >>> perm(Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)))
    Orientation(i=Vector(i=0.0, j=1.0, k=0.0), j=Vector(i=0.0, j=0.0, k=1.0), k=Vector(i=1.0, j=0.0, k=0.0))
    >>> perm(Frame(Vector(1.0, 2.0, 3.0), Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0))))
    Frame(origin=Vector(i=3.0, j=1.0, k=2.0), orientation=Orientation(i=Vector(i=0.0, j=1.0, k=0.0), j=Vector(i=0.0, j=0.0, k=1.0), k=Vector(i=1.0, j=0.0, k=0.0)))
    """  # noqa: E501

    _order: tuple[int, int, int]

    def __init__(self, order: tuple[int, int, int]) -> None:
        self._order = order

    @overload
    def __call__(self, target: Vector) -> Vector:
        ...

    @overload
    def __call__(self, target: Orientation) -> Orientation:
        ...

    @overload
    def __call__(self, target: Frame) -> Frame:
        ...

    @overload
    def __call__(self, target: xp.Array) -> xp.Array:
        ...

    def __call__(self, target: Vector | Orientation | Frame | xp.Array):
        match target:
            case Vector():
                return _permutate_vector(target, self.order)
            case Orientation():
                return _permute_orientation(target, self.order)
            case Frame():
                return _permutate_frame(target, self.order)
            case xp.Array():
                return _permutate_array(target, self.order)
        raise ValueError(f"Unknown data type: {type(target)}")

    @property
    def order(self) -> tuple[int, int, int]:
        return self._order


class Composition(Conversion):
    _lhs: Conversion
    _rhs: Conversion

    def __init__(self, lhs: Conversion, rhs: Conversion) -> None:
        self._lhs = lhs
        self._rhs = rhs

    def __call__(self, target: Vector | Orientation | Frame | xp.Array):
        return self._rhs(self._lhs(target))

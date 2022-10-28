# pyright: reportUnknownMemberType=false

from operator import add, matmul, mul, sub
from typing import Any, Callable

import pytest

import vanilla_roll.array_api as xp
from vanilla_roll.geometry.conversion import Permutation, Transformation
from vanilla_roll.geometry.element import (
    Frame,
    Orientation,
    Vector,
    as_array,
    to_homogeneous,
)
from vanilla_roll.geometry.linalg import normalize_vector


def test_create_vector():
    v = Vector(1.0, 2.0, 3.0)
    assert v.i == 1.0
    assert v.j == 2.0
    assert v.k == 3.0


def test_create_orientation():
    o = Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0))
    assert o.i == Vector(i=1.0, j=0.0, k=0.0)
    assert o.j == Vector(i=0.0, j=1.0, k=0.0)
    assert o.k == Vector(i=0.0, j=0.0, k=1.0)


def test_create_frame():
    o = Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0))
    v = Vector(1.0, 2.0, 3.0)
    f = Frame(v, o)
    assert f.origin == v
    assert f.orientation == o


@pytest.mark.parametrize(
    "lhs, rhs, op, expected",
    [
        (Vector(-1.0, 2.0, 3.0), Vector(1.0, 2.0, 3.0), add, Vector(0.0, 4.0, 6.0)),
        (Vector(-1.0, 2.0, 3.0), Vector(1.0, 2.0, 3.0), sub, Vector(-2.0, 0.0, 0.0)),
        (Vector(-1.0, 2.0, 3.0), 2.0, mul, Vector(-2.0, 4.0, 6.0)),
        (2.0, Vector(-1.0, 2.0, 3.0), mul, Vector(-2.0, 4.0, 6.0)),
        (Vector(-1.0, 2.0, 3.0), Vector(1.0, 2.0, 3.0), matmul, 12.0),
    ],
)
def test_op_vector(
    lhs: Vector | float,
    rhs: Vector | float,
    op: Callable[[Any, Any], Any],
    expected: Vector | float,
):
    assert op(lhs, rhs) == expected


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z",
    [
        (-1.0, 2.0, 3.0),
    ],
)
def test_vector_to_array(x: float, y: float, z: float):
    vector = Vector(x, y, z)
    assert xp.all(as_array(vector) == xp.asarray([z, y, x]))
    assert vector == Vector.of_array(as_array(vector))


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z",
    [
        (3.0, 1.0, float("nan")),
        (3.0, 1.0, float("inf")),
    ],
)
def test_vector_create_fail(x: float, y: float, z: float):
    with pytest.raises(ValueError):
        Vector(x, y, z)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "obj, expected",
    [
        (
            Vector(i=1.0, j=2.0, k=3.0),
            [3.0, 2.0, 1.0],
        ),
        (
            Orientation(
                i=Vector(i=1.0, j=0.0, k=0.0),
                j=Vector(i=0.0, j=1.0, k=0.0),
                k=Vector(i=0.0, j=0.0, k=1.0),
            ),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        (
            Frame(
                Vector(i=1.0, j=2.0, k=3.0),
                Orientation(
                    i=Vector(i=1.0, j=0.0, k=0.0),
                    j=Vector(i=0.0, j=1.0, k=0.0),
                    k=Vector(i=0.0, j=0.0, k=1.0),
                ),
            ),
            [
                [1.0, 0.0, 0.0, 3.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
    ],
)
def test_as_array(
    obj: Vector | Orientation | Frame,
    expected: list[float] | list[list[float]],
    helpers: Any,
):
    actual_array = as_array(obj)
    expected_array = xp.asarray(expected)
    assert actual_array.shape == expected_array.shape

    actual_array = xp.reshape(actual_array, (-1,))
    expected_array = xp.reshape(expected_array, (-1,))
    assert helpers.approx_equal(actual_array, expected_array)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "src, expected",
    [
        (
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 1.0],
        ),
        (
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 1.0]],
        ),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]],
        ),
    ],
)
def test_to_homegeneous(
    src: list[float] | list[list[float]],
    expected: list[float] | list[list[float]],
    helpers: Any,
):
    actual_array = to_homogeneous(xp.asarray(src))
    expected_array = xp.asarray(expected)
    assert actual_array.shape == expected_array.shape

    actual_array = xp.reshape(actual_array, (-1,))
    expected_array = xp.reshape(expected_array, (-1,))
    assert helpers.approx_equal(actual_array, expected_array)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "src_origin, src_orientation, dst_origin, dst_orientation, src_coordinates, expected",
    [
        (
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
        ),
        (
            [1.0, 2.0, 3.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [-1.0, 1.0, 3.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            [[2.0, 1.0, 0.0], [3.0, 3.0, 3.0], [1.0, -1.0, -3.0]],
        ),
        (
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 0.5], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            [[0.0, 0.0, 0.0], [3.0, 2.0, 1.5], [-3.0, -2.0, -1.5]],
        ),
        (
            [0.0, 0.0, 0.0],
            (
                [
                    [8.66025404e-01, 3.53553391e-01, 3.53553391e-01],
                    [-2.77555756e-17, 7.07106781e-01, -7.07106781e-01],
                    [-5.00000000e-01, 6.12372436e-01, 6.12372436e-01],
                ]
            ),
            [0.0, 0.0, 0.0],
            (
                [
                    [8.66025404e-01, 1.29409523e-01, 4.82962913e-01],
                    [-1.38777878e-17, 9.65925826e-01, -2.58819045e-01],
                    [-5.00000000e-01, 2.24143868e-01, 8.36516304e-01],
                ]
            ),
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            [
                [0.0, 0.0, 0.0],
                [-0.1405443698167801, 2.9150636196136475, 2.341506242752075],
                [0.1405443698167801, -2.9150636196136475, -2.341506242752075],
            ],
        ),
    ],
)
def test_transform(
    src_origin: list[float],
    src_orientation: list[list[float]],
    dst_origin: list[float],
    dst_orientation: list[list[float]],
    src_coordinates: list[list[float]],
    expected: Any,
    helpers: Any,
):
    src_coordinate_system = Frame(
        origin=Vector(*src_origin),
        orientation=Orientation(
            i=Vector(*src_orientation[0]),
            j=Vector(*src_orientation[1]),
            k=Vector(*src_orientation[2]),
        ),
    )
    dst_coordinate_system = Frame(
        origin=Vector(*dst_origin),
        orientation=Orientation(
            i=Vector(*dst_orientation[0]),
            j=Vector(*dst_orientation[1]),
            k=Vector(*dst_orientation[2]),
        ),
    )
    transform = Transformation(src_coordinate_system, dst_coordinate_system)

    actual = transform(xp.asarray(src_coordinates).T).T
    expected = xp.asarray(expected)
    assert actual.shape == expected.shape
    assert helpers.approx_equal(actual, expected)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "order, target, expected",
    [
        ((1, 2, 0), Vector(1.0, 2.0, 3.0), Vector(3.0, 1.0, 2.0)),
        (
            (1, 2, 0),
            Orientation(
                Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)
            ),
            Orientation(
                i=Vector(0.0, 1.0, 0.0),
                j=Vector(0.0, 0.0, 1.0),
                k=Vector(1.0, 0.0, 0.0),
            ),
        ),
        (
            (1, 2, 0),
            Frame(
                Vector(1.0, 2.0, 3.0),
                Orientation(
                    Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)
                ),
            ),
            Frame(
                Vector(i=3.0, j=1.0, k=2.0),
                Orientation(
                    i=Vector(i=0.0, j=1.0, k=0.0),
                    j=Vector(i=0.0, j=0.0, k=1.0),
                    k=Vector(i=1.0, j=0.0, k=0.0),
                ),
            ),
        ),
    ],
)
def test_permutation(
    order: tuple[int, int, int],
    target: Vector | Orientation,
    expected: Vector | Orientation,
):
    perm = Permutation(order)
    assert expected == perm(target)
    assert order == perm.order


def test_normalize_vector():
    nv = normalize_vector(Vector(1.0, 2.0, 3.0))
    assert pytest.approx(0.26726, 1e-5) == nv.i
    assert pytest.approx(0.53452, 1e-5) == nv.j
    assert pytest.approx(0.80178, 1e-5) == nv.k

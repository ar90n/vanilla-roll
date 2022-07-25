# pyright: reportUnknownMemberType=false

from typing import Any

import pytest

import vanilla_roll.array_api as xp
from vanilla_roll.geometry import (
    Frame,
    Direction,
    Orientation,
    Point,
    Spacing,
    Transformation,
    Vector,
)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "i, j, k",
    [
        (1.0, 2.0, 3.0),
    ],
)
def test_spacing_create(i: float, j: float, k: float):
    spacing = Spacing(i, j, k)
    assert spacing.i == i
    assert spacing.j == j
    assert spacing.k == k


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "i, j, k",
    [
        (1.0, 2.0, 3.0),
    ],
)
def test_spacing_to_array(i: float, j: float, k: float):
    spacing = Spacing(i, j, k)
    assert xp.all(spacing.to_array() == xp.asarray([k, j, i]))
    assert spacing == Spacing.of_array(spacing.to_array())


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "i, j, k",
    [
        (3.0, 1.0, 0.0),
        (3.0, 1.0, -1.0),
        (3.0, 1.0, float("nan")),
        (3.0, 1.0, float("inf")),
    ],
)
def test_spacing_create_fail(i: float, j: float, k: float):
    with pytest.raises(ValueError):
        Spacing(i, j, k)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z",
    [
        (-1.0, 0.0, 3.0),
    ],
)
def test_point_create(x: float, y: float, z: float):
    point = Point(x, y, z)
    assert point.x == x
    assert point.y == y
    assert point.z == z


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z",
    [
        (-1.0, 0.0, 3.0),
    ],
)
def test_point_to_array(x: float, y: float, z: float):
    point = Point(x, y, z)
    assert xp.all(point.to_array() == xp.asarray([z, y, x]))
    assert point == Point.of_array(point.to_array())


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z",
    [
        (3.0, 1.0, float("nan")),
        (3.0, 1.0, float("inf")),
        (3.0, 1.0, -float("inf")),
    ],
)
def test_point_create_fail(x: float, y: float, z: float):
    with pytest.raises(ValueError):
        Point(x, y, z)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z",
    [
        (-1.0, 2.0, 3.0),
    ],
)
def test_vector_create(x: float, y: float, z: float):
    direction = Vector(x, y, z)
    assert direction.x == x
    assert direction.y == y
    assert direction.z == z


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z",
    [
        (-1.0, 2.0, 3.0),
    ],
)
def test_vector_to_array(x: float, y: float, z: float):
    vector = Vector(x, y, z)
    assert xp.all(vector.to_array() == xp.asarray([z, y, x]))
    assert vector == Vector.of_array(vector.to_array())


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
    "x, y, z, nx, ny, nz",
    [
        (-1.0, 0.0, 3.0, -0.31622776601683794, 0.0, 0.9486832980505138),
    ],
)
def test_direction_create(
    x: float, y: float, z: float, nx: float, ny: float, nz: float
):
    direction = Direction(x, y, z)
    assert direction.x == pytest.approx(nx)
    assert direction.y == pytest.approx(ny)
    assert direction.z == pytest.approx(nz)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z, nx, ny, nz",
    [
        (-1.0, 0.0, 3.0, -0.31622776601683794, 0.0, 0.9486832980505138),
    ],
)
def test_direction_to_array(
    x: float, y: float, z: float, nx: float, ny: float, nz: float, helpers: Any
):
    direction = Direction(x, y, z).to_array()
    assert helpers.approx_equal(direction, xp.asarray([nz, ny, nx]))
    assert helpers.approx_equal(direction, Direction.of_array(direction).to_array())


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "x, y, z",
    [
        (0.0, 0.0, 0.0),
    ],
)
def test_direction_create_fail(x: float, y: float, z: float):
    with pytest.raises(ValueError):
        Direction(x, y, z)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "i, j, k",
    [
        (Direction(1.0, 0.0, 0.0), Direction(0.0, 1.0, 0.0), Direction(0.0, 0.0, 1.0)),
    ],
)
def test_orientation(i: Direction, j: Direction, k: Direction):
    orientation = Orientation(i, j, k)
    assert orientation.k == k
    assert orientation.j == j
    assert orientation.i == i


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "obj, expected",
    [
        (
            Orientation(
                Direction(1.0, 0.0, 0.0),
                Direction(0.0, 1.0, 0.0),
                Direction(0.0, 0.0, 1.0),
            ),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
    ],
)
def test_orientation_to_array(
    obj: Orientation, expected: list[list[float]], helpers: Any
):
    actual_array = obj.to_array()
    expected_array = xp.asarray(expected)
    assert actual_array.shape == expected_array.shape

    actual_array = xp.reshape(actual_array, (-1,))
    expected_array = xp.reshape(expected_array, (-1,))
    assert helpers.approx_equal(actual_array, expected_array)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "src_origin, src_orientation, src_spacing, dst_origin, dst_orientation, dst_spacing, src_coordinates, expected",
    [
        (
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
        ),
        (
            [1.0, 2.0, 3.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 3.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            [[2.0, 1.0, 0.0], [3.0, 3.0, 3.0], [1.0, -1.0, -3.0]],
        ),
        (
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [0.5, 1.0, 3.0],
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
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
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            (
                [
                    [8.66025404e-01, 1.29409523e-01, 4.82962913e-01],
                    [-1.38777878e-17, 9.65925826e-01, -2.58819045e-01],
                    [-5.00000000e-01, 2.24143868e-01, 8.36516304e-01],
                ]
            ),
            [1.0, 1.0, 1.0],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
            [[0.0, 0.0, 0.0], [1.866025, 1.232051, 3.0], [-1.866025, -1.232051, -3.0]],
        ),
    ],
)
def test_transform(
    src_origin: list[float],
    src_orientation: list[list[float]],
    src_spacing: list[float],
    dst_origin: list[float],
    dst_orientation: list[list[float]],
    dst_spacing: list[float],
    src_coordinates: list[list[float]],
    expected: Any,
    helpers: Any,
):
    src_coordinate_system = Frame(
        origin=Point(*src_origin),
        orientation=Orientation(
            i=Direction(*src_orientation[0]),
            j=Direction(*src_orientation[1]),
            k=Direction(*src_orientation[2]),
        ),
        spacing=Spacing(*src_spacing),
    )
    dst_coordinate_system = Frame(
        origin=Point(*dst_origin),
        orientation=Orientation(
            i=Direction(*dst_orientation[0]),
            j=Direction(*dst_orientation[1]),
            k=Direction(*dst_orientation[2]),
        ),
        spacing=Spacing(*dst_spacing),
    )
    transform = Transformation.of(src_coordinate_system, dst_coordinate_system)

    actual = transform.apply(xp.asarray(src_coordinates).T).T
    expected = xp.asarray(expected)
    assert actual.shape == expected.shape
    assert helpers.approx_equal(actual, expected)

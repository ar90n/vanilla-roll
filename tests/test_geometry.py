# pyright: reportUnknownMemberType=false

import pytest

from vanilla_roll.geometry import Direction, Orientation, Origin, Spacing, Spacing2d


@pytest.mark.parametrize(
    "column, row",
    [
        (1.0, 2.0),
    ],
)
def test_spacing2d(column: float, row: float):
    spacing = Spacing2d(column, row)
    assert spacing.column == column
    assert spacing.row == row


@pytest.mark.parametrize(
    "column, row",
    [
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
    ],
)
def test_failure_spacing2d(column: float, row: float):
    with pytest.raises(ValueError):
        Spacing2d(column, row)


@pytest.mark.parametrize(
    "layer, column, row",
    [
        (1.0, 2.0, 3.0),
    ],
)
def test_spacing(layer: float, column: float, row: float):
    spacing = Spacing(layer, column, row)
    assert spacing.column == column
    assert spacing.row == row


@pytest.mark.parametrize(
    "layer, column, row",
    [
        (3.0, 1.0, 0.0),
        (3.0, 1.0, -1.0),
        (3.0, 1.0, float("nan")),
        (3.0, 1.0, float("inf")),
    ],
)
def test_failure_spacing(layer: float, column: float, row: float):
    with pytest.raises(ValueError):
        Spacing(layer, column, row)


@pytest.mark.parametrize(
    "x, y, z",
    [
        (-1.0, 0.0, 3.0),
    ],
)
def test_origin(x: float, y: float, z: float):
    origin = Origin(x, y, z)
    assert origin.x == x
    assert origin.y == y
    assert origin.z == z


@pytest.mark.parametrize(
    "x, y, z",
    [
        (3.0, 1.0, float("nan")),
        (3.0, 1.0, float("inf")),
        (3.0, 1.0, -float("inf")),
    ],
)
def test_failure_origin(x: float, y: float, z: float):
    with pytest.raises(ValueError):
        Origin(x, y, z)


@pytest.mark.parametrize(
    "x, y, z, nx, ny, nz",
    [
        (-1.0, 0.0, 3.0, -0.31622776601683794, 0.0, 0.9486832980505138),
    ],
)
def test_direction(x: float, y: float, z: float, nx: float, ny: float, nz: float):
    direction = Direction(x, y, z)
    assert direction.x == pytest.approx(nx)
    assert direction.y == pytest.approx(ny)
    assert direction.z == pytest.approx(nz)


@pytest.mark.parametrize(
    "x, y, z",
    [
        (0.0, 0.0, 0.0),
        (3.0, 1.0, float("nan")),
        (3.0, 1.0, float("inf")),
    ],
)
def test_failure_direction(x: float, y: float, z: float):
    with pytest.raises(ValueError):
        Direction(x, y, z)


@pytest.mark.parametrize(
    "layer, column, row",
    [
        (Direction(1.0, 0.0, 0.0), Direction(0.0, 1.0, 0.0), Direction(0.0, 0.0, 1.0)),
    ],
)
def test_orientation(layer: Direction, column: Direction, row: Direction):
    orientation = Orientation(layer, column, row)
    assert orientation.layer == layer
    assert orientation.column == column
    assert orientation.row == row

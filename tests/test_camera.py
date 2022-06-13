# pyright: reportUnknownMemberType=false
from typing import Any

import pytest

from vanilla_roll.camera import Camera, Screen
from vanilla_roll.geometry import Direction, Orientation, Point


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "position, forward, up, screen, expected",
    [
        (
            Point(x=0.0, y=0.0, z=0.0),
            Direction(x=0.0, y=0.0, z=1.0),
            Direction(x=0.0, y=-1.0, z=0.0),
            Screen(width=32.0, height=32.0, distance=1.0),
            Point(x=-16.0, y=-16.0, z=1.0),
        ),
        (
            Point(x=12.0, y=-12.0, z=4.0),
            Direction(x=0.0, y=1.0, z=1.0),
            Direction(x=1.0, y=0.0, z=0.0),
            Screen(width=32.0, height=32.0, distance=1.0),
            Point(x=28.0, y=-22.6066, z=16.0208),
        ),
    ],
)
def test_screen_origin(
    position: Point,
    forward: Direction,
    up: Direction,
    screen: Screen,
    expected: Point,
):
    camera = Camera(position=position, forward=forward, up=up, screen=screen)
    actual = camera.screen_origin
    assert actual.x == pytest.approx(expected.x)
    assert actual.y == pytest.approx(expected.y)
    assert actual.z == pytest.approx(expected.z)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "position, forward, up, screen, expected",
    [
        (
            Point(x=0.0, y=0.0, z=0.0),
            Direction(x=0.0, y=0.0, z=1.0),
            Direction(x=0.0, y=-1.0, z=0.0),
            Screen(width=32.0, height=32.0, distance=1.0),
            Point(x=0.0, y=0.0, z=1.0),
        ),
        (
            Point(x=12.0, y=-12.0, z=4.0),
            Direction(x=0.0, y=1.0, z=1.0),
            Direction(x=1.0, y=0.0, z=0.0),
            Screen(width=32.0, height=32.0, distance=1.0),
            Point(x=12.0, y=-11.29289, z=4.707107),
        ),
    ],
)
def test_screen_center(
    position: Point,
    forward: Direction,
    up: Direction,
    screen: Screen,
    expected: Point,
):
    camera = Camera(position=position, forward=forward, up=up, screen=screen)
    actual = camera.screen_center
    assert actual.x == pytest.approx(expected.x)
    assert actual.y == pytest.approx(expected.y)
    assert actual.z == pytest.approx(expected.z)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "position, forward, up, screen, expected",
    [
        (
            Point(x=0.0, y=0.0, z=0.0),
            Direction(x=0.0, y=0.0, z=1.0),
            Direction(x=0.0, y=-1.0, z=0.0),
            Screen(width=32.0, height=32.0, distance=1.0),
            Orientation(
                i=Direction(x=1.0, y=0.0, z=0.0),
                j=Direction(x=0.0, y=1.0, z=0.0),
                k=Direction(x=0.0, y=0.0, z=1.0),
            ),
        ),
        (
            Point(x=12.0, y=-12.0, z=4.0),
            Direction(x=0.0, y=1.0, z=1.0),
            Direction(x=1.0, y=0.0, z=0.0),
            Screen(width=32.0, height=32.0, distance=1.0),
            Orientation(
                i=Direction(x=0.0, y=1.0, z=-1.0),
                j=Direction(x=-1.0, y=0.0, z=0.0),
                k=Direction(x=0.0, y=1.0, z=1.0),
            ),
        ),
    ],
)
def test_screen_orientation(
    position: Point,
    forward: Direction,
    up: Direction,
    screen: Screen,
    expected: Point,
    helpers: Any,
):
    camera = Camera(position=position, forward=forward, up=up, screen=screen)
    actual = camera.screen_orientation
    assert helpers.approx_equal(actual.to_array(), expected.to_array())


@pytest.mark.parametrize(
    "width, height, distance",
    [
        (0.0, 32.0, 1.0),
        (-32.0, 32.0, 1.0),
        (32.0, 0.0, 1.0),
        (32.0, -32.0, 1.0),
        (32.0, 32.0, -1.0),
        (float("nan"), 32.0, 1.0),
        (float("inf"), 32.0, 1.0),
        (32.0, float("nan"), 1.0),
        (32.0, float("inf"), 1.0),
        (32.0, 32.0, float("nan")),
        (32.0, 32.0, float("inf")),
    ],
)
def test_screen_create_fail(width: float, height: float, distance: float):
    with pytest.raises(ValueError):
        Screen(width=width, height=height, distance=distance)

# pyright: reportUnknownMemberType=false
from typing import Any

import pytest

import vanilla_roll.array_api as xp
from tests.conftest import Helpers
from vanilla_roll.anatomy_orientation import AnatomyAxis, Axial, Sagittal
from vanilla_roll.camera import Camera, ViewVolume, create_from_anatomy_axis
from vanilla_roll.geometry.element import Frame, Orientation, Vector, as_array
from vanilla_roll.geometry.linalg import normalize_vector


def _calc_orientation(forward: Vector, up: Vector) -> Orientation:
    j = Vector.of_array(-as_array(normalize_vector(up)))
    k = normalize_vector(forward)
    i = Vector.of_array(xp.linalg.cross(as_array(k), as_array(j)))
    return Orientation(i=i, j=j, k=k)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "position, forward, up, view_volume, expected",
    [
        (
            Vector(i=0.0, j=0.0, k=0.0),
            Vector(i=0.0, j=0.0, k=1.0),
            Vector(i=0.0, j=-1.0, k=0.0),
            ViewVolume(width=32.0, height=32.0, near=1.0, far=10.0),
            Vector(i=-16.0, j=-16.0, k=1.0),
        ),
        (
            Vector(i=12.0, j=-12.0, k=4.0),
            Vector(i=0.0, j=1.0, k=1.0),
            Vector(i=1.0, j=0.0, k=0.0),
            ViewVolume(width=32.0, height=32.0, near=1.0, far=10.0),
            Vector(i=28.0, j=-22.6066, k=16.0208),
        ),
    ],
)
def test_screen_origin(
    position: Vector,
    forward: Vector,
    up: Vector,
    view_volume: ViewVolume,
    expected: Vector,
):
    frame = Frame(position, _calc_orientation(forward, up))
    camera = Camera(frame, view_volume=view_volume)
    actual = camera.screen_origin
    assert actual.i == pytest.approx(expected.i)
    assert actual.j == pytest.approx(expected.j)
    assert actual.k == pytest.approx(expected.k)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "position, forward, up, view_volume, expected",
    [
        (
            Vector(i=0.0, j=0.0, k=0.0),
            Vector(i=0.0, j=0.0, k=1.0),
            Vector(i=0.0, j=-1.0, k=0.0),
            ViewVolume(width=32.0, height=32.0, near=1.0, far=1.0),
            Vector(i=0.0, j=0.0, k=1.0),
        ),
        (
            Vector(i=12.0, j=-12.0, k=4.0),
            Vector(i=0.0, j=1.0, k=1.0),
            Vector(i=1.0, j=0.0, k=0.0),
            ViewVolume(width=32.0, height=32.0, near=1.0, far=1.0),
            Vector(i=12.0, j=-11.29289, k=4.707107),
        ),
    ],
)
def test_screen_center(
    position: Vector,
    forward: Vector,
    up: Vector,
    view_volume: ViewVolume,
    expected: Vector,
):
    frame = Frame(position, _calc_orientation(forward, up))
    camera = Camera(frame, view_volume=view_volume)
    actual = camera.screen_center
    assert actual.i == pytest.approx(expected.i)
    assert actual.j == pytest.approx(expected.j)
    assert actual.k == pytest.approx(expected.k)


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
        ViewVolume(width=width, height=height, near=distance, far=distance)


@pytest.mark.parametrize(
    "volume_params, face, up, expected",
    [
        (
            {},
            Sagittal.ANTERIOR,
            Axial.SUPERIOR,
            Camera(
                Frame(
                    Vector(i=128.0, j=349.7025033688163, k=128.0),
                    Orientation(
                        i=Vector(i=-1.0, j=0.0, k=0.0),
                        j=Vector(i=0.0, j=0.0, k=-1.0),
                        k=Vector(i=0.0, j=-1.0, k=0.0),
                    ),
                ),
                ViewVolume(
                    443.40500673763256,
                    443.40500673763256,
                    near=0,
                    far=443.40500673763256,
                ),
            ),
        )
    ],
)
def test_create_aligned_camera(
    helpers: Helpers,
    volume_params: dict[str, Any],
    face: AnatomyAxis,
    up: AnatomyAxis,
    expected: Camera,
):
    volume = helpers.create_volume(**volume_params)
    actual = create_from_anatomy_axis(volume, face=face, up=up)

    assert helpers.approx_equal(
        as_array(actual.frame.origin), as_array(expected.frame.origin)
    )
    assert helpers.approx_equal(
        as_array(actual.frame.orientation), as_array(expected.frame.orientation)
    )
    assert actual.view_volume.far == pytest.approx(expected.view_volume.far, abs=1e-12)
    assert actual.view_volume.near == pytest.approx(
        expected.view_volume.near, abs=1e-12
    )
    assert actual.view_volume.width == pytest.approx(
        expected.view_volume.width, abs=1e-12
    )
    assert actual.view_volume.height == pytest.approx(
        expected.view_volume.height, abs=1e-12
    )

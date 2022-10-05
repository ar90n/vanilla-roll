from typing import Iterator

from vanilla_roll.anatomy_orientation import (
    CSA,
    AnatomyAxis,
    AnatomyOrientation,
    Axial,
    Coronal,
    Sagittal,
    get_direction,
)
from vanilla_roll.camera import create_from_anatomy_axis
from vanilla_roll.camera_sequence import create_circular
from vanilla_roll.geometry.element import Orientation, Vector, as_array
from vanilla_roll.rendering import create_renderer
from vanilla_roll.rendering.algorithm import ShearWarp
from vanilla_roll.rendering.mode import Average, Mode
from vanilla_roll.rendering.projection import Orthogoal
from vanilla_roll.rendering.types import RenderingResult
from vanilla_roll.volume import Volume


def _get_default_up_axis(anatomy_axis: AnatomyAxis) -> AnatomyAxis:
    match anatomy_axis:
        case Axial():
            return Sagittal.ANTERIOR
        case _:
            return Axial.SUPERIOR


def render(
    target: Volume,
    face: AnatomyAxis = Sagittal.ANTERIOR,
    mode: Mode = Average(),
    spacing: float | None = None,
) -> RenderingResult:
    camera = create_from_anatomy_axis(target, face=face, up=_get_default_up_axis(face))
    renderer = create_renderer(
        target,
        projection=Orthogoal(),
        rendering_method=mode,
        algorithm=ShearWarp(),
    )
    return renderer(camera, spacing=spacing)


def render_horizontal_rotations(
    target: Volume, mode: Mode = Average(), n: int = 16, spacing: float | None = None
) -> Iterator[RenderingResult]:
    yield from _render_rotations(
        target,
        rotation_axis=Axial.SUPERIOR,
        initial_axis=Sagittal.ANTERIOR,
        mode=mode,
        n=n,
        spacing=spacing,
        up_rot=1.0,
    )


def render_vertical_rotations(
    target: Volume, mode: Mode = Average(), n: int = 16, spacing: float | None = None
) -> Iterator[RenderingResult]:
    yield from _render_rotations(
        target,
        rotation_axis=Coronal.RIGHT,
        initial_axis=Sagittal.ANTERIOR,
        mode=mode,
        n=n,
        spacing=spacing,
        up_rot=0.0,
    )


def _get_direction_in_world(
    orientation: Orientation,
    anatomy_orientation: AnatomyOrientation,
    anatomy_axis: AnatomyAxis,
) -> Vector:
    tmp = as_array(orientation) @ as_array(
        get_direction(anatomy_orientation, anatomy_axis)
    )
    return Vector.of_array(tmp)


def _render_rotations(
    target: Volume,
    rotation_axis: AnatomyAxis,
    initial_axis: AnatomyAxis,
    up_rot: float,
    mode: Mode,
    n: int,
    spacing: float | None,
) -> Iterator[RenderingResult]:
    renderer = create_renderer(
        target,
        projection=Orthogoal(),
        rendering_method=mode,
        algorithm=ShearWarp(),
    )
    anatomy_orientation = (
        target.anatomy_orientation
        if target.anatomy_orientation is not None
        else CSA(k=Axial.INFERIOR, j=Sagittal.ANTERIOR, i=Coronal.LEFT)
    )

    axis = _get_direction_in_world(
        target.frame.orientation, anatomy_orientation, rotation_axis
    )
    initial = _get_direction_in_world(
        target.frame.orientation, anatomy_orientation, initial_axis
    )
    for camera in create_circular(
        target, n=n, axis=axis, initial=initial, up_rot=up_rot
    ):
        yield renderer(camera, spacing=spacing)

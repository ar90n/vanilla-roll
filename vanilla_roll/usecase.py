from typing import Iterator

from vanilla_roll.anatomy_orientation import (
    CSA,
    AnatomyAxis,
    Axial,
    Coronal,
    Sagittal,
    get_direction,
)
from vanilla_roll.camera import create_from_anatomy_axis
from vanilla_roll.camera_sequence import create_circular
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
    target: Volume, face: AnatomyAxis = Sagittal.ANTERIOR, mode: Mode = Average()
) -> RenderingResult:
    camera = create_from_anatomy_axis(target, face=face, up=_get_default_up_axis(face))
    renderer = create_renderer(
        target,
        projection=Orthogoal(),
        rendering_method=mode,
        algorithm=ShearWarp(),
    )
    return renderer(camera)


def render_horizontal_rotations(
    target: Volume, mode: Mode = Average(), n: int = 16
) -> Iterator[RenderingResult]:
    yield from _render_rotations(target, Axial.SUPERIOR, mode, n)


def render_vertical_rotations(
    target: Volume, mode: Mode = Average(), n: int = 16
) -> Iterator[RenderingResult]:
    yield from _render_rotations(target, Coronal.LEFT, mode, n)


def _render_rotations(
    target: Volume, rotation_axis: AnatomyAxis, mode: Mode, n: int
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

    axis = get_direction(anatomy_orientation, rotation_axis)
    up = get_direction(anatomy_orientation, _get_default_up_axis(rotation_axis))
    for camera in create_circular(target, n=n, axis=axis, up=up):
        yield renderer(camera)

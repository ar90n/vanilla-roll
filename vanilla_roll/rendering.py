from dataclasses import dataclass
from typing import Callable, TypeAlias

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
from vanilla_roll.camera import Camera
from vanilla_roll.geometry import (
    CoordinateSystem,
    Orientation,
    Point,
    Spacing,
    Transform,
    world_coordinate_system,
)
from vanilla_roll.volume import Volume


@dataclass(frozen=True)
class Orthogoal:
    pass


@dataclass
class Perspective:
    pass


Projection = Orthogoal | Perspective


@dataclass
class MPR:
    sampling_method: xpe.SamplingMethod = "linear"


@dataclass
class MIP:
    pass


@dataclass
class VR:
    pass


Rendering = MPR | MIP | VR


Warp: TypeAlias = Callable[
    [Volume, Camera, Projection], tuple[Volume, Camera, Projection]
]


def _direciton_aligned_linspace(
    direction: xp.Array,
    /,
    length: float,
    samples: int,
    transform: Transform | None = None,
):
    space = xp.reshape(direction, (-1, 1)) * xp.linspace(
        -length / 2, length / 2, samples
    )
    if transform is not None:
        space = transform.apply(space)
    return space


def _calc_screen_coordinates_in_volume_space(
    volume_coordinate_system: CoordinateSystem,
    camera: Camera,
    columns: int,
    rows: int,
) -> xp.Array:
    transform = Transform.of(src=world_coordinate_system, dst=volume_coordinate_system)

    row_space = _direciton_aligned_linspace(
        camera.screen_orientation.i.to_array(),
        length=camera.screen.width,
        samples=columns,
        transform=transform,
    )
    column_space = _direciton_aligned_linspace(
        camera.screen_orientation.j.to_array(),
        length=camera.screen.height,
        samples=rows,
        transform=transform,
    )
    center = transform.apply(camera.screen_center.to_array())

    coords = (
        center
        + xp.reshape(row_space.T, (1, -1, 3))
        + xp.reshape(column_space.T, (-1, 1, 3))
    )
    return xp.reshape(coords, (-1, 3))


def _create_rendering_region_mask(volume: Volume, coords: xp.Array) -> xp.Array:
    min_coords = xp.zeros(3)
    max_coords = xp.astype(xp.asarray(volume.data.shape), xp.float64)
    return xp.all(
        xp.logical_and(min_coords <= coords, coords < max_coords),
        axis=-1,
    )


def _create_mpr_image(
    volume: Volume,
    /,
    camera: Camera,
    columns: int,
    rows: int,
    sampling_method: xpe.SamplingMethod,
) -> xp.Array:
    coords = _calc_screen_coordinates_in_volume_space(
        volume.coordinate_system, camera, columns, rows
    )

    mask = _create_rendering_region_mask(volume, coords)
    samples = xp.zeros(rows * columns)
    samples[mask] = xpe.sample(
        volume.data, coordinates=coords[mask], method=sampling_method
    )
    return xp.reshape(samples, (rows, columns))


@dataclass(frozen=True)
class RenderingResult:
    image: xp.Array
    spacing: Spacing | None
    origin: Point | None
    orientation: Orientation | None


def render(
    columns: int,
    rows: int,
    volume: Volume,
    camera: Camera,
    projection: Projection,
    rendering: Rendering,
    transform: Warp | None = None,
) -> RenderingResult:
    if transform is not None:
        volume, camera, projection = transform(volume, camera, projection)

    match (projection, rendering):
        case (_, MPR(sampling_method)):
            return RenderingResult(
                image=_create_mpr_image(
                    volume,
                    camera=camera,
                    columns=columns,
                    rows=rows,
                    sampling_method=sampling_method,
                ),
                spacing=Spacing(
                    i=camera.screen.width / columns,
                    j=camera.screen.height / rows,
                    k=volume.coordinate_system.spacing.k,
                ),
                origin=camera.screen_origin,
                orientation=camera.screen_orientation,
            )
        case _:
            raise NotImplementedError(f"{projection} {rendering}")

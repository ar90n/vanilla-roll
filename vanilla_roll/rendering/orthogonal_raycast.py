import math
from typing import Callable

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
from vanilla_roll.camera import Camera
from vanilla_roll.geometry.element import (
    Vector,
    world_frame,
    as_array,
)
from vanilla_roll.geometry.linalg import normalize_vector
from vanilla_roll.geometry.conversion import Transformation
from vanilla_roll.volume import Volume
from vanilla_roll.rendering.types import RenderingResult, Renderer
from vanilla_roll.rendering.accumulation import Accumulator


def _direciton_aligned_linspace(
    direction: xp.Array,
    /,
    start: float,
    end: float,
    samples: int,
):
    return xp.reshape(direction, (-1, 1)) * xp.linspace(start, end, samples)


def _calc_orthogonal_view_volume_coordinates(
    transform: Transformation,
    camera: Camera,
    columns: int,
    rows: int,
    depth: int,
) -> xp.Array:
    row_direction = normalize_vector(camera.screen_orientation.i)
    row_space = _direciton_aligned_linspace(
        as_array(row_direction),
        start=-(camera.view_volume.width / 2.0),
        end=(camera.view_volume.width / 2.0),
        samples=columns,
    )
    row_space = transform(row_space)

    column_direction = normalize_vector(camera.screen_orientation.j)
    column_space = _direciton_aligned_linspace(
        as_array(column_direction),
        start=-(camera.view_volume.height / 2.0),
        end=(camera.view_volume.height / 2.0),
        samples=rows,
    )
    column_space = transform(column_space)

    depth_direction = normalize_vector(camera.screen_orientation.k)
    depth_space = _direciton_aligned_linspace(
        as_array(depth_direction),
        start=camera.view_volume.near,
        end=camera.view_volume.far,
        samples=depth,
    )
    depth_space = transform(depth_space)

    origin = transform(as_array(camera.frame.origin))

    coords = (
        origin
        + xp.reshape(row_space.T, (1, 1, -1, 3))
        + xp.reshape(column_space.T, (1, -1, 1, 3))
        + xp.reshape(depth_space.T, (-1, 1, 1, 3))
    )
    return xp.reshape(coords, (-1, 3))


def _create_mask(coords: xp.Array, /, shape: tuple[int, int, int]) -> xp.Array:
    return xp.all(
        xp.logical_and(
            xp.zeros(3) <= coords,
            coords < xp.asarray(shape, dtype=xp.float64),
        ),
        axis=-1,
    )


def _extract_orthogonal_view_volume(
    volume: Volume,
    /,
    camera: Camera,
    columns: int,
    rows: int,
    layers: int,
    sampling_method: xpe.SamplingMethod,
) -> xp.Array:
    transform = Transformation(src=world_frame, dst=volume.frame)
    coords = _calc_orthogonal_view_volume_coordinates(
        transform, camera, columns, rows, layers
    )

    mask = _create_mask(coords, shape=volume.data.shape)
    samples = xp.zeros(math.prod(mask.shape))
    samples[mask] = xpe.sample(
        volume.data, coordinates=coords[mask], method=sampling_method
    )
    return xp.reshape(samples, (layers, rows, columns))


def _calc_layers(camera: Camera, step: float) -> int:
    return max(
        1, int(math.floor((camera.view_volume.far - camera.view_volume.near) / step))
    )


def create_renderer(
    volume: Volume,
    /,
    step: float,
    accumulator_constructor: Callable[[tuple[int, int]], Accumulator],
    sampling_method: xpe.SamplingMethod,
) -> Renderer:
    def _render(camera: Camera, shape: tuple[int, int]) -> RenderingResult:
        rows, columns = shape
        layers = _calc_layers(camera, step)
        view_volume = _extract_orthogonal_view_volume(
            volume,
            camera=camera,
            columns=columns,
            rows=rows,
            layers=layers,
            sampling_method=sampling_method,
        )

        accumulator = accumulator_constructor(shape)
        for k in range(layers):
            accumulator.add(view_volume[k, :, :], step)

        return RenderingResult(
            image=accumulator.get_result(),
            spacing=Vector(
                i=camera.view_volume.width / columns,
                j=camera.view_volume.height / rows,
                k=step,
            ),
            origin=camera.screen_origin,
            orientation=camera.screen_orientation,
        )

    return _render

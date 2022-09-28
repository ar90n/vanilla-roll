import math
from typing import Callable

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
from vanilla_roll.camera import Camera
from vanilla_roll.geometry.conversion import Transformation
from vanilla_roll.geometry.element import Vector, as_array, world_frame
from vanilla_roll.geometry.linalg import normalize_vector
from vanilla_roll.rendering.composition import Composer
from vanilla_roll.rendering.image_proc import resize_image
from vanilla_roll.rendering.types import Image, Renderer, RenderingResult
from vanilla_roll.volume import Volume


def _direciton_aligned_linspace(
    direction: xp.Array,
    /,
    start: float,
    end: float,
    samples: int,
):
    return xp.reshape(direction, (-1, 1)) * xp.linspace(start, end, samples)


def _calc_orthogonal_view_volume_coordinates(
    transform: Transformation, camera: Camera, step: float
) -> tuple[xp.Array, tuple[int, int, int]]:
    columns = int(camera.view_volume.width / step)
    rows = int(camera.view_volume.height / step)
    depth = _calc_layers(camera, step)

    row_direction = normalize_vector(camera.screen_orientation.i)
    row_space = _direciton_aligned_linspace(
        as_array(row_direction),
        start=-(camera.view_volume.width / 2.0),
        end=(camera.view_volume.width / 2.0),
        samples=columns,
    )

    column_direction = normalize_vector(camera.screen_orientation.j)
    column_space = _direciton_aligned_linspace(
        as_array(column_direction),
        start=-(camera.view_volume.height / 2.0),
        end=(camera.view_volume.height / 2.0),
        samples=rows,
    )

    depth_direction = normalize_vector(camera.screen_orientation.k)
    depth_space = _direciton_aligned_linspace(
        as_array(depth_direction),
        start=camera.view_volume.near,
        end=camera.view_volume.far,
        samples=depth,
    )

    coords = (
        as_array(camera.frame.origin)
        + xp.reshape(row_space.T, (1, 1, -1, 3))
        + xp.reshape(column_space.T, (1, -1, 1, 3))
        + xp.reshape(depth_space.T, (-1, 1, 1, 3))
    )
    coords = transform(xp.reshape(coords, (-1, 3)).T).T
    return coords, (depth, rows, columns)


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
    step: float,
    sampling_method: xpe.SamplingMethod,
) -> xp.Array:
    transform = Transformation(src=world_frame, dst=volume.frame)
    coords, shape = _calc_orthogonal_view_volume_coordinates(transform, camera, step)

    mask = _create_mask(coords, shape=volume.data.shape)
    samples = xp.zeros(math.prod(mask.shape))
    samples[mask] = xpe.sample(
        volume.data, coordinates=coords[mask], method=sampling_method
    )
    return xp.reshape(samples, shape)


def _calc_layers(camera: Camera, step: float) -> int:
    return max(
        1, int(math.floor((camera.view_volume.far - camera.view_volume.near) / step))
    )


def _compose_view_volume_voxels(
    view_volume_voxels: xp.Array,
    accumulator_constructor: Callable[[tuple[int, int]], Composer],
    spacing: float,
) -> Image:
    layers, rows, columns = view_volume_voxels.shape
    accumulator = accumulator_constructor((rows, columns))
    for k in range(layers):
        accumulator.add(view_volume_voxels[k, :, :], spacing)
    return accumulator.compose()


def create_renderer(
    volume: Volume,
    /,
    step: float,
    accumulator_constructor: Callable[[tuple[int, int]], Composer],
    sampling_method: xpe.SamplingMethod,
) -> Renderer:
    def _render(camera: Camera, spacing: float | None = None) -> RenderingResult:
        if spacing is None:
            spacing = step

        shape = (
            int(camera.view_volume.height / spacing),
            int(camera.view_volume.width / spacing),
        )

        view_volume_voxels = _extract_orthogonal_view_volume(
            volume,
            camera=camera,
            step=step,
            sampling_method=sampling_method,
        )

        composed_image = _compose_view_volume_voxels(
            view_volume_voxels, accumulator_constructor, step
        )

        return RenderingResult(
            image=resize_image(composed_image, shape, method=sampling_method),
            spacing=Vector(i=spacing, j=spacing, k=step),
            origin=camera.screen_origin,
            orientation=camera.screen_orientation,
        )

    return _render

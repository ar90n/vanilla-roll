import math
from typing import Callable, Generic, Protocol, TypeVar

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
from vanilla_roll.anatomy_orientation import create as create_anatomy_orientation
from vanilla_roll.camera import Camera
from vanilla_roll.geometry.conversion import (
    Composition,
    Conversion,
    Permutation,
    Transformation,
)
from vanilla_roll.geometry.element import (
    Frame,
    Orientation,
    Vector,
    as_array,
    world_frame,
)
from vanilla_roll.geometry.linalg import norm, normalize_vector
from vanilla_roll.rendering.composition import Composer, Slice2d
from vanilla_roll.rendering.image_proc import affine_image
from vanilla_roll.rendering.types import Image, Renderer, RenderingResult
from vanilla_roll.volume import Volume

T = TypeVar("T")


class HasIJK(Protocol, Generic[T]):
    i: T
    j: T
    k: T


def _permutate_kji(obj: HasIJK[T], order: tuple[int, int, int]) -> tuple[T, T, T]:
    k, j, i = (
        [
            obj.k,
            obj.j,
            obj.i,
        ][i]
        for i in order
    )
    return (k, j, i)


def _create_transformation(
    src: Frame, dst: Frame
) -> tuple[Transformation, Transformation]:
    src_to_dst = Transformation(src, dst)
    dst_to_src = Transformation(dst, src)
    return src_to_dst, dst_to_src


def _find_principal_viewing_axis(viewing_direction: Vector) -> int:
    return int(xp.argmax(xp.abs(as_array(viewing_direction))))


def _create_permutation_from_principal_axis(
    viewing_direction: Vector,
) -> tuple[Permutation, Permutation]:
    principal_axis = _find_principal_viewing_axis(viewing_direction)
    match principal_axis:
        case 0:
            return (
                Permutation((0, 1, 2)),
                Permutation((0, 1, 2)),
            )
        case 1:
            return (
                Permutation((1, 2, 0)),
                Permutation((2, 0, 1)),
            )
        case 2:
            return (
                Permutation((2, 0, 1)),
                Permutation((1, 2, 0)),
            )
        case int() as x:
            raise ValueError(f"Invalid principal axis: {x}")


def _permute_volume(target: Volume, order: tuple[int, int, int]) -> Volume:
    k, j, i = _permutate_kji(target.frame.orientation, order)
    permutated_frame = Frame(
        orientation=Orientation(
            i=i,
            j=j,
            k=k,
        ),
        origin=target.frame.origin,
    )

    permutated_anatomy_orientation = target.anatomy_orientation
    if permutated_anatomy_orientation is not None:
        k, j, i = _permutate_kji(permutated_anatomy_orientation, order)
        permutated_anatomy_orientation = create_anatomy_orientation(i=i, j=j, k=k)

    return Volume(
        data=xp.permute_dims(target.data, order),
        frame=permutated_frame,
        anatomy_orientation=permutated_anatomy_orientation,
    )


def _calc_shearing(viewing_direction: Vector) -> Vector:
    si = -viewing_direction.i / viewing_direction.k
    sj = -viewing_direction.j / viewing_direction.k
    return Vector(i=si, j=sj, k=0.0)


def _calc_translation(shearing_distance: Vector, shape: tuple[int, int]) -> Vector:
    ti = -shearing_distance.i * shape[0] if shearing_distance.i < 0 else 0
    tj = -shearing_distance.j * shape[0] if shearing_distance.j < 0 else 0
    return Vector(i=ti, j=tj, k=0)


def _calc_intermediate_image_shape(
    shearing: Vector, translation: Vector, shape: tuple[int, int, int]
) -> tuple[int, int]:
    max_depth = shape[0]
    width = (
        int(math.ceil(translation.i) + max(0.0, math.floor(shearing.i * max_depth)))
        + shape[2]
    )
    height = (
        int(math.ceil(translation.j) + max(0.0, math.floor(shearing.j * max_depth)))
        + shape[1]
    )
    return (height, width)


def _create_direction_mat_in_world_frame(
    camera: Camera, inv_conversion: Conversion
) -> xp.Array:
    inv_orientation = inv_conversion(camera.frame.orientation)

    k = normalize_vector(inv_orientation.k)
    j = normalize_vector(inv_orientation.j)
    i = normalize_vector(inv_orientation.i)
    return xp.asarray([[k.k, j.k, i.k], [k.j, j.j, i.j], [k.i, j.i, i.i]])


def _mesh_grid_from_origin(
    origin: Vector, shape: tuple[int, int]
) -> tuple[xp.Array, xp.Array]:
    iss, jss = xp.meshgrid(xp.arange(shape[1]), xp.arange(shape[0]), indexing="xy")
    iss = xp.astype(iss, xp.float32) - origin.i
    jss = xp.astype(jss, xp.float32) - origin.j
    return jss, iss


def _create_viewv_volume_region(camera: Camera) -> tuple[xp.Array, xp.Array]:
    slab_length = camera.view_volume.far - camera.view_volume.near
    half_width = camera.view_volume.width / 2
    half_height = camera.view_volume.height / 2
    min_point = xp.asarray([0, -half_height, -half_width])
    max_point = xp.asarray([slab_length, half_height, half_width])
    return min_point, max_point


def _calc_update_region_slice(
    i: int, shearing: Vector, translation: Vector, shape: tuple[int, int]
) -> Slice2d:
    ox = int(math.floor(shearing.i * i) + math.ceil(translation.i))
    oy = int(math.floor(shearing.j * i) + math.ceil(translation.j))
    return Slice2d(j=slice(oy, oy + shape[0]), i=slice(ox, ox + shape[1]))


def _get_slice_indices(volume: Volume, camera: Camera) -> range:
    return (
        range(volume.data.shape[0])
        if 0 < camera.forward.k
        else range(volume.data.shape[0] - 1, -1, -1)
    )


def _render_intermediate_image(
    perm_volume: Volume,
    shearing: Vector,
    translation: Vector,
    perm_camera: Camera,
    inv_conversion: Conversion,
    accumulator_constructor: Callable[[tuple[int, int]], Composer],
) -> Image:
    dir_mat = _create_direction_mat_in_world_frame(perm_camera, inv_conversion)

    grid_j, grid_i = _mesh_grid_from_origin(
        perm_camera.frame.origin, perm_volume.data.shape[1:]
    )
    min_point, max_point = _create_viewv_volume_region(perm_camera)

    def _create_mask(idx: int, shape: tuple[int, int]) -> xp.Array:
        grid_k = idx * xp.ones(shape, dtype=xp.float32) - perm_camera.frame.origin.k
        grid = xp.stack([grid_k, grid_j, grid_i], axis=2)
        grid_in_world = xp.reshape(
            inv_conversion(xp.reshape(grid, (-1, 3)).T).T, grid.shape
        )
        points_in_view = grid_in_world @ dir_mat
        masks = (min_point < points_in_view) & (points_in_view < max_point)
        return masks[:, :, 0] & masks[:, :, 1] & masks[:, :, 2]

    accumulator = accumulator_constructor(
        _calc_intermediate_image_shape(shearing, translation, perm_volume.data.shape)
    )
    thickness = norm(perm_volume.frame.orientation.k)

    for i in _get_slice_indices(perm_volume, perm_camera):
        s = xp.astype(perm_volume.data[i, :, :], xp.float64)
        mask = _create_mask(i, s.shape)
        slice = _calc_update_region_slice(i, shearing, translation, s.shape)
        accumulator.add(s, thickness, mask=mask, slice=slice)
    return accumulator.compose()


def _calc_warp_matrix(
    view_mat: xp.Array,
    translation: Vector,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> xp.Array:
    align_mat = xp.asarray(
        [
            [dst_shape[0] / src_shape[0], 0, dst_shape[0] / 2],
            [0, dst_shape[1] / src_shape[1], dst_shape[1] / 2],
            [0, 0, 1],
        ],
        dtype=view_mat.dtype,
    )
    translate_mat = xp.asarray(
        [
            [1, 0, -translation.j],
            [0, 1, -translation.i],
            [0, 0, 1],
        ],
        dtype=view_mat.dtype,
    )
    warp_mat = align_mat @ view_mat[1:, 1:] @ translate_mat
    return warp_mat


def _warp(
    intermediate_image: Image,
    view_matrix: xp.Array,
    translation: Vector,
    warped_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    sampling_method: xpe.SamplingMethod,
) -> Image:
    warp_mat = _calc_warp_matrix(view_matrix, translation, warped_shape, dst_shape)
    return affine_image(
        intermediate_image, xp.linalg.inv(warp_mat.T), dst_shape, method=sampling_method
    )


def _calc_warped_shape(camera: Camera) -> tuple[int, int]:
    return (
        int(camera.screen_shape[0] / norm(camera.frame.orientation.j)),
        int(camera.screen_shape[1] / norm(camera.frame.orientation.i)),
    )


def _apply_conversion(camera: Camera, conversion: Conversion) -> Camera:
    return Camera(
        frame=conversion(camera.frame),
        view_volume=camera.view_volume,
    )


def create_renderer(
    volume: Volume,
    /,
    accumulator_constructor: Callable[[tuple[int, int]], Composer],
    sampling_method: xpe.SamplingMethod,
) -> Renderer:
    to_volume = Transformation(src=world_frame, dst=volume.frame)
    rotate_to_volume, inv_rotate_to_volume = _create_transformation(
        src=Frame(
            orientation=world_frame.orientation,
            origin=volume.frame.origin,
        ),
        dst=volume.frame,
    )

    def _render(camera: Camera, spacing: float | None = None) -> RenderingResult:
        if spacing is None:
            spacing = 1.0

        shape = (
            int(camera.view_volume.height / spacing),
            int(camera.view_volume.width / spacing),
        )

        viewing_direction = rotate_to_volume(camera.screen_orientation).k
        perm, inv_perm = _create_permutation_from_principal_axis(viewing_direction)
        inv_conversion = Composition(inv_perm, inv_rotate_to_volume)

        perm_camera = _apply_conversion(_apply_conversion(camera, to_volume), perm)
        perm_volume = _permute_volume(volume, perm.order)
        perm_viewing_direction = perm(viewing_direction)

        shearing = _calc_shearing(perm_viewing_direction)
        translation = _calc_translation(shearing, perm_volume.data.shape)

        intermediate_image = _render_intermediate_image(
            perm_volume,
            shearing,
            translation,
            perm_camera,
            inv_conversion,
            accumulator_constructor,
        )

        result_image = _warp(
            intermediate_image,
            view_matrix=perm_camera.view_matrix,
            translation=translation,
            warped_shape=_calc_warped_shape(camera),
            dst_shape=shape,
            sampling_method=sampling_method,
        )

        return RenderingResult(
            image=result_image,
            spacing=Vector(i=spacing, j=spacing, k=perm_volume.spacing.k),
            origin=camera.screen_origin,
            orientation=camera.screen_orientation,
        )

    return _render

import math
from typing import Callable

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
import vanilla_roll.array_api_image as xpi
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
from vanilla_roll.rendering.types import (
    ColorImage,
    Image,
    MonoImage,
    Renderer,
    RenderingResult,
)
from vanilla_roll.volume import Volume


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
    k, j, i = (
        [
            target.frame.orientation.k,
            target.frame.orientation.j,
            target.frame.orientation.i,
        ][i]
        for i in order
    )
    return Volume(
        data=xp.permute_dims(target.data, order),
        frame=Frame(
            orientation=Orientation(
                i=i,
                j=j,
                k=k,
            ),
            origin=target.frame.origin,
        ),
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
    k = as_array(normalize_vector(inv_conversion(camera.frame.orientation.k)))
    j = as_array(normalize_vector(inv_conversion(camera.frame.orientation.j)))
    i = as_array(normalize_vector(inv_conversion(camera.frame.orientation.i)))

    return xp.asarray([k, j, i]).T


def _mesh_grid_from_origin(
    origin: Vector, shape: tuple[int, int]
) -> tuple[xp.Array, xp.Array]:
    iss, jss = xp.meshgrid(xp.arange(shape[1]), xp.arange(shape[0]))
    iss = xp.astype(iss, xp.float64) - origin.i
    jss = xp.astype(jss, xp.float64) - origin.j
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
        grid_k = idx * xp.ones(shape, dtype=xp.float64) - perm_camera.frame.origin.k
        grid = xp.stack([grid_k, grid_j, grid_i], axis=2)
        grid_in_world = xp.reshape(
            inv_conversion(xp.reshape(grid, (-1, 3)).T).T, grid.shape
        )
        points_in_world = grid_in_world @ dir_mat
        masks = (min_point < points_in_world) & (points_in_world < max_point)
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
    dst_shape: tuple[int, int],
    src_shape: tuple[int, int],
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
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> Image:
    warp_mat = _calc_warp_matrix(view_matrix, translation, dst_shape, src_shape)
    match intermediate_image:
        case MonoImage(l):
            warped_l = xpi.affine_transform(l, xp.linalg.inv(warp_mat.T), dst_shape)
            return MonoImage(l=warped_l)
        case ColorImage(r, g, b):
            warped_r = xpi.affine_transform(r, xp.linalg.inv(warp_mat.T), dst_shape)
            warped_g = xpi.affine_transform(g, xp.linalg.inv(warp_mat.T), dst_shape)
            warped_b = xpi.affine_transform(b, xp.linalg.inv(warp_mat.T), dst_shape)
            return ColorImage(r=warped_r, g=warped_g, b=warped_b)


def create_renderer(
    volume: Volume,
    /,
    accumulator_constructor: Callable[[tuple[int, int]], Composer],
    sampling_method: xpe.SamplingMethod,
) -> Renderer:
    transform, inv_transform = _create_transformation(src=world_frame, dst=volume.frame)

    def _render(camera: Camera, shape: tuple[int, int]) -> RenderingResult:
        viewing_direction = transform(camera.screen_orientation.k)
        perm, inv_perm = _create_permutation_from_principal_axis(viewing_direction)
        inv_conversion = Composition(inv_perm, inv_transform)

        perm_camera = camera.apply(transform).apply(perm)
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
            perm_camera.view_matrix,
            translation,
            perm_camera.screen_shape,
            shape,
        )
        result_spacing = Vector(
            i=camera.view_volume.width / shape[1],
            j=camera.view_volume.height / shape[0],
            k=0.0,
        )
        return RenderingResult(
            image=result_image,
            spacing=result_spacing,
            origin=camera.screen_origin,
            orientation=camera.screen_orientation,
        )

    return _render

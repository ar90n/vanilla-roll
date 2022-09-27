import math
from typing import Iterator

import vanilla_roll.array_api as xp
from vanilla_roll.camera import Camera, create_lookat_center
from vanilla_roll.geometry.conversion import Transformation
from vanilla_roll.geometry.element import Vector, world_frame
from vanilla_roll.geometry.linalg import norm
from vanilla_roll.volume import Volume


def calc_rot_mat_i(sin: float) -> xp.Array:
    cos = (1.0 - sin**2) ** 0.5
    return xp.asarray(
        [
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def calc_rot_mat_j(sin: float) -> xp.Array:
    cos = (1.0 - sin**2) ** 0.5
    return xp.asarray(
        [
            [cos, 0.0, -sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [sin, 0.0, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def calc_rot_mat_k(sin: float) -> xp.Array:
    cos = (1.0 - sin**2) ** 0.5
    return xp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, -sin, 0.0],
            [0.0, sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def create_circular(
    target: Volume, axis: Vector, up: Vector, n: int = 16
) -> Iterator[Camera]:
    sin_i = axis.j / (axis.j**2 + axis.k**2) ** 0.5
    rot_i = calc_rot_mat_i(sin_i)

    sin_j = axis.i / norm(axis)
    rot_j = calc_rot_mat_j(sin_j)

    sin_k = math.sin(2 * math.pi / n)
    rot_k = calc_rot_mat_k(sin_k)

    rot = rot_i @ rot_j

    to_world = Transformation(target.frame, world_frame)
    center = to_world(
        Vector(
            k=target.shape[0] / 2.0, j=target.shape[1] / 2.0, i=target.shape[2] / 2.0
        )
    )

    radius = target.diagonal_length / 2.0
    initial_position = xp.asarray([0.0, 0.0, radius, 1.0])
    for _ in range(n):
        p = Vector.of_array((rot @ initial_position)[:3]) + center
        yield create_lookat_center(target, position=p, up=up)
        rot = rot @ rot_k

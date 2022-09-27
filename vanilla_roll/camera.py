from dataclasses import dataclass
from functools import cached_property

import vanilla_roll.array_api as xp
from vanilla_roll.anatomy_orientation import AnatomyAxis, get_direction
from vanilla_roll.geometry.conversion import Transformation
from vanilla_roll.geometry.element import (
    Frame,
    Orientation,
    Vector,
    as_array,
    world_frame,
)
from vanilla_roll.geometry.linalg import normalize_vector
from vanilla_roll.validation import (
    IsFinite,
    IsGreaterEqualThan,
    IsGreaterThan,
    Validator,
)
from vanilla_roll.volume import Volume


@dataclass(frozen=True)
class ViewVolume:
    width: float
    height: float
    far: float
    near: float

    def __post_init__(self) -> None:
        size_validator = Validator(rules=[IsGreaterThan(0.0), IsFinite()])
        for f in ("width", "height"):
            if exception := size_validator(f, getattr(self, f)):
                raise exception

        depth_validator = Validator(rules=[IsGreaterEqualThan(0.0), IsFinite()])
        for f in ("far", "near"):
            if exception := depth_validator(f, getattr(self, f)):
                raise exception

        if self.far < self.near:
            raise ValueError(
                f"far ({self.far}) must be greater than near ({self.near})"
            )


@dataclass(frozen=True)
class Camera:
    frame: Frame
    view_volume: ViewVolume

    @property
    def forward(self) -> Vector:
        return self.frame.orientation.k

    @cached_property
    def up(self) -> Vector:
        return Vector.of_array(-as_array(self.frame.orientation.j))

    @cached_property
    def screen_origin(self) -> Vector:
        dir_i = as_array(normalize_vector(self.screen_orientation.i))
        dir_j = as_array(normalize_vector(self.screen_orientation.j))

        return Vector.of_array(
            as_array(self.screen_center)
            - (self.view_volume.width / 2.0) * dir_i
            - (self.view_volume.height / 2.0) * dir_j
        )

    @cached_property
    def screen_center(self) -> Vector:
        center = as_array(self.frame.origin) + self.view_volume.near * as_array(
            self.forward
        )
        return Vector.of_array(center)

    @property
    def screen_shape(self) -> tuple[int, int]:
        return (int(self.view_volume.height), int(self.view_volume.width))

    @property
    def screen_orientation(self) -> Orientation:
        return self.frame.orientation

    @cached_property
    def screen_frame(self) -> Frame:
        return Frame(
            origin=self.screen_origin,
            orientation=self.screen_orientation,
        )

    @cached_property
    def view_matrix(self) -> xp.Array:
        return xp.linalg.inv(as_array(self.frame))


def _calc_orientation(forward: Vector, up: Vector) -> Orientation:
    j = Vector.of_array(-as_array(normalize_vector(up)))
    k = normalize_vector(forward)
    i = Vector.of_array(xp.linalg.cross(as_array(k), as_array(j)))
    return Orientation(i=i, j=j, k=k)


def _center_of(shape: tuple[int, int, int]) -> Vector:
    return Vector(k=shape[0] / 2.0, j=shape[1] / 2.0, i=shape[2] / 2.0)


def _calc_diag_length(shape: tuple[int, int, int]) -> float:
    return (shape[0] ** 2 + shape[1] ** 2 + shape[2] ** 2) ** 0.5


def _create_default_view_volume(target: Volume) -> ViewVolume:
    diag_length = _calc_diag_length(target.data.shape)
    scale = target.diagonal_length / diag_length
    return ViewVolume(
        width=scale * diag_length,
        height=scale * diag_length,
        far=scale * diag_length,
        near=0,
    )


def create_from_volume_coordinates(
    target: Volume,
    /,
    *,
    position: Vector,
    forward: Vector,
    up: Vector,
    view_volume: ViewVolume | None = None,
) -> Camera:
    if view_volume is None:
        view_volume = _create_default_view_volume(target)

    to_world = Transformation(target.frame, world_frame)
    to_world2 = Transformation(
        target.frame,
        Frame(orientation=world_frame.orientation, origin=target.frame.origin),
    )

    return Camera(
        frame=Frame(
            origin=to_world(position),
            orientation=_calc_orientation(to_world2(forward), to_world2(up)),
        ),
        view_volume=view_volume,
    )


def create_from_anatomy_axis(
    target: Volume,
    /,
    *,
    face: AnatomyAxis,
    up: AnatomyAxis,
    view_volume: ViewVolume | None = None,
) -> Camera:
    if target.anatomy_orientation is None:
        raise ValueError("target.anatomy_orientation is None")

    if view_volume is None:
        view_volume = _create_default_view_volume(target)

    half_diag_length = _calc_diag_length(target.data.shape) / 2.0
    position = _center_of(target.data.shape) + half_diag_length * get_direction(
        target.anatomy_orientation, face
    )

    return create_from_volume_coordinates(
        target,
        position=position,
        forward=get_direction(target.anatomy_orientation, face.inverse()),
        up=get_direction(target.anatomy_orientation, up),
        view_volume=view_volume,
    )


def create_lookat_center(
    target: Volume,
    /,
    *,
    position: Vector,
    up: Vector,
    view_volume: ViewVolume | None = None,
) -> Camera:
    if view_volume is None:
        view_volume = _create_default_view_volume(target)

    center = Transformation(target.frame, world_frame)(_center_of(target.data.shape))
    forward = normalize_vector(center - position)

    return Camera(
        frame=Frame(origin=position, orientation=_calc_orientation(forward, up)),
        view_volume=view_volume,
    )

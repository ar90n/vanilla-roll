# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import Any, cast

import metaimageio  # type: ignore
import numpy as np
import numpy.typing as npt

import vanilla_roll.array_api as xp
from vanilla_roll.geometry.element import Frame, Vector, Orientation, as_array
from vanilla_roll.volume import Volume


def _get_spacing_from_mha_meta(meta: dict[str, Any]) -> Vector:
    return Vector(
        i=meta["ElementSpacing"][0],
        j=meta["ElementSpacing"][1],
        k=meta["ElementSpacing"][2],
    )


def _get_origin_from_mha_meta(meta: dict[str, Any]) -> Vector:
    return Vector(
        i=meta["Offset"][0],
        j=meta["Offset"][1],
        k=meta["Offset"][2],
    )


def _get_orientation_from_mha_meta(meta: dict[str, Any]) -> Orientation:
    row_dir = Vector(
        i=meta["TransformMatrix"][0, 0],
        j=meta["TransformMatrix"][0, 1],
        k=meta["TransformMatrix"][0, 2],
    )
    column_dir = Vector(
        i=meta["TransformMatrix"][1, 0],
        j=meta["TransformMatrix"][1, 1],
        k=meta["TransformMatrix"][1, 2],
    )
    layler_dir = Vector(
        i=meta["TransformMatrix"][2, 0],
        j=meta["TransformMatrix"][2, 1],
        k=meta["TransformMatrix"][2, 2],
    )
    return Orientation(k=layler_dir, j=column_dir, i=row_dir)


def read_mha(path: str | Path) -> Volume:
    np_array_data, meta = cast(
        tuple[npt.NDArray[Any], dict[str, Any]], metaimageio.read(path)
    )

    data = xp.from_dlpack(np_array_data.astype(np.int16))
    spacing = _get_spacing_from_mha_meta(meta)
    origin = _get_origin_from_mha_meta(meta)
    orientation = _get_orientation_from_mha_meta(meta)
    scaled_orientation = Orientation(
        i=Vector.of_array(spacing.i * as_array(orientation.i)),
        j=Vector.of_array(spacing.j * as_array(orientation.j)),
        k=Vector.of_array(spacing.k * as_array(orientation.k)),
    )
    return Volume(
        data,
        frame=Frame(origin, scaled_orientation),
    )

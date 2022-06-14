# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import Any, cast

import metaimageio  # type: ignore
import numpy as np
import numpy.typing as npt

import vanilla_roll.array_api as xp
from vanilla_roll.geometry import (
    CoordinateSystem,
    Direction,
    Orientation,
    Point,
    Spacing,
)
from vanilla_roll.volume import Volume


def _get_spacing_from_mha_meta(meta: dict[str, Any]) -> Spacing:
    return Spacing(
        i=meta["ElementSpacing"][0],
        j=meta["ElementSpacing"][1],
        k=meta["ElementSpacing"][2],
    )


def _get_origin_from_mha_meta(meta: dict[str, Any]) -> Point:
    return Point(
        z=meta["Offset"][2],
        y=meta["Offset"][1],
        x=meta["Offset"][0],
    )


def _get_orientation_from_mha_meta(meta: dict[str, Any]) -> Orientation:
    row_dir = Direction(
        x=meta["TransformMatrix"][0, 0],
        y=meta["TransformMatrix"][0, 1],
        z=meta["TransformMatrix"][0, 2],
    )
    column_dir = Direction(
        x=meta["TransformMatrix"][1, 0],
        y=meta["TransformMatrix"][1, 1],
        z=meta["TransformMatrix"][1, 2],
    )
    layler_dir = Direction(
        x=meta["TransformMatrix"][2, 0],
        y=meta["TransformMatrix"][2, 1],
        z=meta["TransformMatrix"][2, 2],
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
    return Volume(
        data,
        coordinate_system=CoordinateSystem(origin, orientation, spacing),
    )

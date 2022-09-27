# pyright: reportUnknownMemberType=false

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import vanilla_roll.array_api as xp
from vanilla_roll.anatomy_orientation import AnatomyOrientation
from vanilla_roll.anatomy_orientation import parse as parse_anatomy_orientation
from vanilla_roll.geometry.element import Frame, Orientation, Vector, as_array
from vanilla_roll.volume import Volume

try:
    import metaimageio  # type: ignore
    import numpy as np
    import numpy.typing as npt

    HAS_METAIMAGEIO = True
except ImportError:
    HAS_METAIMAGEIO = False  # type: ignore
    if TYPE_CHECKING:
        import metaimageio  # type: ignore
        import numpy as np
        import numpy.typing as npt


@dataclass(frozen=True)
class MHAIOParams:
    force_lps: bool = True


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


def _get_anatomy_orientation(
    meta: dict[str, Any], params: MHAIOParams
) -> AnatomyOrientation | None:
    if params.force_lps:
        return parse_anatomy_orientation("LPS")

    try:
        return parse_anatomy_orientation(meta["AnatomicalOrientation"])
    except (KeyError, ValueError):
        return None


def read_mha(path: str | Path, params: MHAIOParams = MHAIOParams()) -> Volume:
    if not HAS_METAIMAGEIO:
        raise RuntimeError("metaimageio is not installed")

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
    anatomy_orientation = _get_anatomy_orientation(meta, params)
    return Volume(
        data,
        frame=Frame(origin, scaled_orientation),
        anatomy_orientation=anatomy_orientation,
    )

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import vanilla_roll.array_api as xp
from vanilla_roll.anatomy_orientation import AnatomyOrientation
from vanilla_roll.anatomy_orientation import parse as parse_anatomy_orientation
from vanilla_roll.geometry.element import Frame, Orientation, Vector, as_array
from vanilla_roll.volume import Volume

try:
    import nibabel as nib
    import numpy as np
    import numpy.typing as npt
    from nibabel.orientations import aff2axcodes
    from nibabel.spatialimages import SpatialImage

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False  # type: ignore
    if TYPE_CHECKING:
        import nibabel as nib
        import numpy as np
        import numpy.typing as npt
        from nibabel.orientations import aff2axcodes
        from nibabel.spatialimages import SpatialImage


@dataclass(frozen=True)
class NIFTIIOParams:
    pass


def _get_spacing(nii: SpatialImage) -> Vector:
    zooms: tuple[float, float, float] = nii.header.get_zooms()  # type: ignore
    return Vector(
        i=zooms[0],
        j=zooms[1],
        k=zooms[2],
    )


def _get_origin(nii: SpatialImage) -> Vector:
    translates: npt.NDArray[np.float64] = nii.affine[:3, 3]
    return Vector(
        i=translates[0],
        j=translates[1],
        k=translates[2],
    )


def _get_orientation(nii: SpatialImage) -> Orientation:
    orientation: npt.NDArray[np.float64] = nii.affine[:3, :3]
    return Orientation(
        i=Vector.of_array(orientation[0]),
        j=Vector.of_array(orientation[1]),
        k=Vector.of_array(orientation[2]),
    )


def _get_anatomy_orientation(nii: SpatialImage) -> AnatomyOrientation:
    axcodes: tuple[str] = aff2axcodes(nii.affine)
    return parse_anatomy_orientation("".join(axcodes[::-1]))


def _get_data(nii: SpatialImage) -> xp.Array:
    return xp.from_dlpack(nii.get_data())


def _load_data(path: str | Path) -> SpatialImage:
    image: SpatialImage = nib.load(path)
    return cast(SpatialImage, image)


def read_nifti(path: str | Path, params: NIFTIIOParams = NIFTIIOParams()) -> Volume:
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel is not installed")

    nii = _load_data(path)

    data = _get_data(nii)
    spacing = _get_spacing(nii)
    origin = _get_origin(nii)
    orientation = _get_orientation(nii)
    scaled_orientation = Orientation(
        i=Vector.of_array(spacing.i * as_array(orientation.i)),
        j=Vector.of_array(spacing.j * as_array(orientation.j)),
        k=Vector.of_array(spacing.k * as_array(orientation.k)),
    )
    anatomy_orientation = _get_anatomy_orientation(nii)
    return Volume(
        data,
        frame=Frame(origin, scaled_orientation),
        anatomy_orientation=anatomy_orientation,
    )

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import vanilla_roll.array_api as xp
from vanilla_roll.anatomy_orientation import AnatomyOrientation
from vanilla_roll.anatomy_orientation import parse as parse_anatomy_orientation
from vanilla_roll.geometry.element import Frame, Orientation, Vector, as_array
from vanilla_roll.volume import Volume

try:
    import nibabel as nib  # type: ignore
    import numpy as np
    import numpy.typing as npt
    from nibabel.orientations import aff2axcodes  # type: ignore

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False  # type: ignore
    if TYPE_CHECKING:
        import nibabel as nib
        import numpy as np
        import numpy.typing as npt
        from nibabel.orientations import aff2axcodes  # type: ignore


@dataclass(frozen=True)
class NIFTIIOParams:
    pass


def _get_spacing(nii: nib.Nifti1Image) -> Vector:
    zooms: tuple[float, float, float] = nii.header.get_zooms()  # type: ignore
    return Vector(
        i=zooms[0],
        j=zooms[1],
        k=zooms[2],
    )


def _get_origin(nii: nib.Nifti1Image) -> Vector:
    translates: npt.NDArray[np.float64] = nii.affine[:3, 3]  # type: ignore
    return Vector(
        i=translates[0],
        j=translates[1],
        k=translates[2],
    )


def _get_orientation(nii: nib.Nifti1Image) -> Orientation:
    orientation: npt.NDArray[np.float64] = nii.affine[:3, :3]  # type: ignore
    return Orientation(
        i=Vector.of_array(orientation[0]),
        j=Vector.of_array(orientation[1]),
        k=Vector.of_array(orientation[2]),
    )


def _get_anatomy_orientation(nii: nib.Nifti1Image) -> AnatomyOrientation:
    axcodes: tuple[str] = aff2axcodes(nii.affine)  # type: ignore
    return parse_anatomy_orientation("".join(axcodes[::-1]))  # type: ignore


def _get_data(nii: nib.Nifti1Image) -> xp.Array:
    return xp.from_dlpack(nii.get_data())  # type: ignore


def read_nifti(path: str | Path, params: NIFTIIOParams = NIFTIIOParams()) -> Volume:
    if not HAS_NIBABEL:
        raise RuntimeError("metaimageio is not installed")

    nii: nib.Nifti1Image = nib.load(path)  # type: ignore

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

# pyright: reportUnknownMemberType=false

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import vanilla_roll.array_api as xp
from vanilla_roll.anatomy_orientation import CSA, Axial, Coronal, Sagittal
from vanilla_roll.geometry.element import Frame, Orientation, Vector
from vanilla_roll.geometry.linalg import normalize_vector
from vanilla_roll.volume import Volume

try:
    import numpy as np
    import pydicom  # type: ignore
    import pydicom.dicomdir

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False  # type: ignore
    if TYPE_CHECKING:
        import numpy as np
        import pydicom
        import pydicom.dicomdir


@dataclass(frozen=True)
class SliceIntervalStat:
    max: float
    min: float
    avg: float


@dataclass(frozen=True)
class DicomIOParams:
    acceptable_slice_interval_error: float = 0.05


def read_dicom(
    paths: Iterable[str | Path], params: DicomIOParams = DicomIOParams()
) -> Volume:
    if not HAS_PYDICOM:
        raise RuntimeError("pydicom is not installed")

    def _calc_slice_interval_stats(
        sorted_dcms: list[pydicom.FileDataset],
    ) -> SliceIntervalStat:
        zs = xp.asarray([d.ImagePositionPatient[2] for d in sorted_dcms])
        slice_intervals = zs[1:] - zs[:-1]
        max_interval = float(xp.max(slice_intervals))
        min_interval = float(xp.min(slice_intervals))
        avg_interval = float(xp.mean(slice_intervals))
        return SliceIntervalStat(max=max_interval, min=min_interval, avg=avg_interval)

    def _has_same_attributes(dcms: list[pydicom.FileDataset], attr: str) -> bool:
        return all(getattr(d, attr) == getattr(dcms[0], attr) for d in dcms)

    def _has_uniform_slice_interval(
        interval_stat: SliceIntervalStat, acceptable_slice_interval_error: float
    ) -> bool:
        return (interval_stat.max - interval_stat.min) < acceptable_slice_interval_error

    def _is_orthogonal_volume(
        sorted_dcms: list[pydicom.FileDataset], criteria: float = 1e-6
    ) -> bool:
        base_dcm = sorted_dcms[0]
        orientation = _calc_orientation(base_dcm, 1.0)
        base_dir_i = normalize_vector(orientation.i)
        base_dir_j = normalize_vector(orientation.j)
        base_dir_k = normalize_vector(orientation.k)

        if criteria < abs(base_dir_i @ base_dir_j):
            return False

        base_position = Vector(
            i=float(base_dcm.ImagePositionPatient[0]),
            j=float(base_dcm.ImagePositionPatient[1]),
            k=float(base_dcm.ImagePositionPatient[2]),
        )
        for dcm in sorted_dcms[1:]:
            cur_position = Vector(
                i=float(dcm.ImagePositionPatient[0]),
                j=float(dcm.ImagePositionPatient[1]),
                k=float(dcm.ImagePositionPatient[2]),
            )

            cur_dir_k = normalize_vector(cur_position - base_position)
            if abs(base_dir_k @ cur_dir_k) < (1.0 - criteria):
                return False
        return True

    def _calc_orientation(dcm: pydicom.FileDataset, interval: float) -> Orientation:
        axis_i = np.array(dcm.ImageOrientationPatient[:3], dtype=np.float64)
        axis_j = np.array(dcm.ImageOrientationPatient[3:], dtype=np.float64)
        axis_k = np.cross(axis_j, axis_i)

        axis_i *= float(dcm.PixelSpacing[0])
        axis_j *= float(dcm.PixelSpacing[1])
        axis_k *= interval

        return Orientation(
            i=Vector(i=axis_i[0], j=axis_i[1], k=axis_i[2]),
            j=Vector(i=axis_j[0], j=axis_j[1], k=axis_j[2]),
            k=Vector(i=axis_k[0], j=axis_k[1], k=axis_k[2]),
        )

    def _validate(
        dcms: list[pydicom.FileDataset],
        interval_stat: SliceIntervalStat,
        acceptable_slice_interval_error: float,
    ) -> None:
        if not _has_uniform_slice_interval(
            interval_stat, acceptable_slice_interval_error
        ):
            raise ValueError(
                f"Slice interval is not uniform: min={interval_stat.min}, max={interval_stat.max}, avg={interval_stat.avg}"  # noqa: E501
            )

        if not _is_orthogonal_volume(dcms):
            raise ValueError("Slice orientation is not orthogonal")

        attrs = [
            "ImageOrientationPatient",
            "PixelSpacing",
            "Rows",
            "Columns",
            "SeriesInstanceUID",
            "StudyInstanceUID",
        ]
        for attr in attrs:
            if not _has_same_attributes(dcms, attr):
                raise ValueError(f"Attribute {attr} is not uniform")

    def _array_from_dcm(dcm: pydicom.FileDataset) -> xp.Array:
        pixel_array = pydicom.pixel_data_handlers.apply_rescale(dcm.pixel_array, dcm)  # type: ignore
        return xp.from_dlpack(pixel_array)  # type: ignore

    dcms: list[pydicom.FileDataset] = []
    for p in paths:
        d = pydicom.dcmread(p)
        if isinstance(d, pydicom.dicomdir.DicomDir):
            raise ValueError(f"DicomDir is not supported: {str(p)}")
        dcms.append(d)
    dcms.sort(key=lambda d: -d.ImagePositionPatient[2])

    if len(dcms) == 0:
        raise ValueError("No dicom files are found")

    interval_stat = _calc_slice_interval_stats(dcms)
    _validate(dcms, interval_stat, params.acceptable_slice_interval_error)

    data = xp.stack([_array_from_dcm(d) for d in dcms], axis=0)
    return Volume(
        data=data,
        frame=Frame(
            origin=Vector(
                i=float(dcms[0].ImagePositionPatient[0]),
                j=float(dcms[0].ImagePositionPatient[1]),
                k=float(dcms[0].ImagePositionPatient[2]),
            ),
            orientation=_calc_orientation(dcms[0], interval_stat.avg),
        ),
        anatomy_orientation=CSA(Coronal.LEFT, Sagittal.POSTERIOR, Axial.SUPERIOR),
    )

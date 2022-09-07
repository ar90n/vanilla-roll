from dataclasses import dataclass
from typing import Protocol

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
from vanilla_roll.rendering.transfer_function import TransferFunction
from vanilla_roll.rendering.types import ColorImage, Image, MonoImage


@dataclass(frozen=True)
class Slice2d:
    i: slice
    j: slice


class Composer(Protocol):
    def add(
        self,
        image: xp.Array,
        thickness: float,
        /,
        *,
        mask: xp.Array | None = None,
        slice: Slice2d | None = None,
    ) -> None:
        ...

    def compose(self) -> Image:
        ...


class AccMax(Composer):
    _sampling_method: xpe.SamplingMethod
    _accumulation: xp.Array

    def __init__(
        self,
        shape: tuple[int, int],
        /,
        *,
        sampling_method: xpe.SamplingMethod = "linear",
    ) -> None:
        self._sampling_method = sampling_method
        self._accumulation = -float("inf") * xp.ones(shape, dtype=xp.float64)

    def add(
        self,
        image: xp.Array,
        thickness: float,
        /,
        *,
        mask: xp.Array | None = None,
        slice: Slice2d | None = None,
    ) -> None:
        if image.ndim != 2:
            raise ValueError("image must be 2d")

        acc_region = (
            self._accumulation
            if slice is None
            else self._accumulation[slice.j, slice.i]
        )

        max_mask = acc_region < image
        if mask is not None:
            max_mask = max_mask & mask
        acc_region[max_mask] = image[max_mask]

    def compose(self) -> Image:
        none_value_mask = xp.isinf(self._accumulation)
        luma = self._accumulation[:, :]
        luma[none_value_mask] = 0
        return MonoImage(l=luma)


class AccMin(Composer):
    _sampling_method: xpe.SamplingMethod
    _accumulation: xp.Array

    def __init__(
        self,
        shape: tuple[int, int],
        /,
        *,
        sampling_method: xpe.SamplingMethod = "linear",
    ) -> None:
        self._sampling_method = sampling_method
        self._accumulation = float("inf") * xp.ones(shape, dtype=xp.float64)

    def add(
        self,
        image: xp.Array,
        thickness: float,
        /,
        *,
        mask: xp.Array | None = None,
        slice: Slice2d | None = None,
    ) -> None:
        if image.ndim != 2:
            raise ValueError("image must be 2d")

        acc_region = (
            self._accumulation
            if slice is None
            else self._accumulation[slice.j, slice.i]
        )

        min_mask = image < acc_region
        if mask is not None:
            min_mask = min_mask & mask
        acc_region[min_mask] = image[min_mask]

    def compose(self) -> Image:
        none_value_mask = xp.isinf(self._accumulation)
        luma = self._accumulation[:, :]
        luma[none_value_mask] = 0
        return MonoImage(l=luma)


class AccMean(Composer):
    _sampling_method: xpe.SamplingMethod
    _accumulation: xp.Array
    _acc_count: int

    def __init__(
        self,
        shape: tuple[int, int],
        /,
        *,
        sampling_method: xpe.SamplingMethod = "linear",
    ) -> None:
        self._sampling_method = sampling_method
        self._accumulation = xp.zeros(shape, dtype=xp.float64)
        self._acc_count = 0

    def add(
        self,
        image: xp.Array,
        thickness: float,
        /,
        *,
        mask: xp.Array | None = None,
        slice: Slice2d | None = None,
    ) -> None:
        if image.ndim != 2:
            raise ValueError("image must be 2d")

        acc_region = (
            self._accumulation
            if slice is None
            else self._accumulation[slice.j, slice.i]
        )

        self._acc_count += 1
        if mask is not None:
            acc_region[mask] += image[mask]
        else:
            acc_region += image

    def compose(self) -> Image:
        if self._acc_count == 0:
            return MonoImage(xp.zeros(self._accumulation.shape, dtype=xp.float64))

        luma = self._accumulation / self._acc_count
        return MonoImage(l=luma)


class AccVR(Composer):
    _sampling_method: xpe.SamplingMethod
    _acc_r: xp.Array
    _acc_g: xp.Array
    _acc_b: xp.Array
    _acc_alpha: xp.Array
    _transfer_function: TransferFunction

    def __init__(
        self,
        shape: tuple[int, int],
        transfer_function: TransferFunction,
        /,
        *,
        sampling_method: xpe.SamplingMethod = "linear",
    ) -> None:
        self._sampling_method = sampling_method
        self._acc_r = xp.zeros(shape, dtype=xp.float64)
        self._acc_g = xp.zeros(shape, dtype=xp.float64)
        self._acc_b = xp.zeros(shape, dtype=xp.float64)
        self._acc_alpha = xp.zeros(shape, dtype=xp.float64)
        self._transfer_function = transfer_function

    def add(
        self,
        image: xp.Array,
        thickness: float,
        /,
        *,
        mask: xp.Array | None = None,
        slice: Slice2d | None = None,
    ) -> None:
        if image.ndim != 2:
            raise ValueError("image must be 2d")

        acc_r_region = self._acc_r if slice is None else self._acc_r[slice.j, slice.i]
        acc_g_region = self._acc_g if slice is None else self._acc_g[slice.j, slice.i]
        acc_b_region = self._acc_b if slice is None else self._acc_b[slice.j, slice.i]
        acc_alpha_region = (
            self._acc_alpha if slice is None else self._acc_alpha[slice.j, slice.i]
        )

        ret = self._transfer_function(image)
        roi_alpha = 1.0 - xp.exp(-ret.opacity * thickness)
        if mask is not None:
            cur_alpha = (1.0 - acc_alpha_region[mask]) * roi_alpha[mask]
            acc_r_region[mask] += cur_alpha * ret.r[mask]
            acc_g_region[mask] += cur_alpha * ret.g[mask]
            acc_b_region[mask] += cur_alpha * ret.b[mask]
            acc_alpha_region[mask] += cur_alpha
        else:
            cur_alpha = (1.0 - acc_alpha_region) * roi_alpha
            acc_r_region += cur_alpha * ret.r
            acc_g_region += cur_alpha * ret.g
            acc_b_region += cur_alpha * ret.b
            acc_alpha_region += cur_alpha

    def compose(self) -> Image:
        return ColorImage(r=self._acc_r, g=self._acc_g, b=self._acc_b)

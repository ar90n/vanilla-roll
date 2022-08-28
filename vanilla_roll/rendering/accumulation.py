from dataclasses import dataclass
from typing import Protocol

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe


@dataclass(frozen=True)
class Slice2d:
    i: slice
    j: slice


class Accumulator(Protocol):
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

    def get_result(self) -> xp.Array:
        ...


class AccMax(Accumulator):
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

    def get_result(self) -> xp.Array:
        none_value_mask = xp.isinf(self._accumulation)
        ret = self._accumulation[:, :]
        ret[none_value_mask] = 0
        return ret


class AccMin(Accumulator):
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

    def get_result(self) -> xp.Array:
        none_value_mask = xp.isinf(self._accumulation)
        ret = self._accumulation[:, :]
        ret[none_value_mask] = 0
        return ret


class AccMean(Accumulator):
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

    def get_result(self) -> xp.Array:
        if self._acc_count == 0:
            return xp.zeros(self._accumulation.shape, dtype=xp.float64)
        return self._accumulation / self._acc_count

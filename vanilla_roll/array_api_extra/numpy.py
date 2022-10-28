from typing import Any

import numpy.typing as npt

import vanilla_roll.array_api as xp

from .type import Number


def _get_raw_array(array: xp.Array) -> npt.NDArray[Any]:
    raw_array = getattr(array, "_array", None)
    if raw_array is None:
        raise ValueError(f"Expected numpy array, got {type(array)}")
    return raw_array


def take(array: xp.Array, /, *, indices: xp.Array) -> xp.Array:
    raw_array = _get_raw_array(array)
    raw_indices = _get_raw_array(indices)
    return xp.asarray(raw_array.reshape(-1)[raw_indices])


def put(array: xp.Array, /, *, indices: xp.Array, values: xp.Array) -> None:
    raw_array = _get_raw_array(array)
    raw_indices = _get_raw_array(indices)
    raw_values = _get_raw_array(values)
    raw_array.reshape(-1)[raw_indices] = raw_values


def clip(
    array: xp.Array,
    /,
    *,
    a_min: xp.Array | Number | None,
    a_max: xp.Array | Number | None = None,
) -> xp.Array:
    raw_array = _get_raw_array(array)
    raw_a_min = _get_raw_array(a_min) if isinstance(a_min, xp.Array) else a_min
    raw_a_max = _get_raw_array(a_max) if isinstance(a_max, xp.Array) else a_max
    return xp.asarray(raw_array.clip(raw_a_min, raw_a_max))  # type: ignore


def asnumpy(array: xp.Array) -> npt.NDArray[Any]:
    return _get_raw_array(array)

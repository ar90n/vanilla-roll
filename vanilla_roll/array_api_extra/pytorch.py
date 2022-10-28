from typing import Any, cast

import numpy.typing as npt
import torch

import vanilla_roll.array_api as xp

from .type import Number


def _get_raw_array(array: xp.Array) -> torch.Tensor:
    return cast(torch.Tensor, array)


def take(array: xp.Array, /, *, indices: xp.Array) -> xp.Array:
    raw_array = cast(torch.Tensor, array)
    raw_indices = cast(torch.Tensor, indices)
    return xp.asarray(torch.take(raw_array, raw_indices))


def put(array: xp.Array, /, *, indices: xp.Array, values: xp.Array) -> None:
    raw_array = cast(torch.Tensor, array)
    raw_indices = cast(torch.Tensor, indices)
    raw_values = cast(torch.Tensor, values)
    raw_array.put_(raw_indices, raw_values)


def clip(
    array: xp.Array,
    /,
    *,
    a_min: xp.Array | Number | None,
    a_max: xp.Array | Number | None = None,
) -> xp.Array:
    def _to_raw_limit(
        min_or_max: xp.Array | Number | None,
    ) -> torch.Tensor | None:
        if min_or_max is None:
            return None
        if isinstance(min_or_max, xp.Array):
            return cast(torch.Tensor, min_or_max)
        return torch.tensor(min_or_max)

    raw_array = cast(torch.Tensor, array)
    raw_a_min = _to_raw_limit(a_min)
    raw_a_max = _to_raw_limit(a_max)

    return xp.asarray(torch.clamp(raw_array, min=raw_a_min, max=raw_a_max))


def asnumpy(array: xp.Array) -> npt.NDArray[Any]:
    raw_array = _get_raw_array(array)
    return raw_array.to("cpu").detach().numpy()

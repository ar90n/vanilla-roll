import itertools
from typing import TYPE_CHECKING, Sequence

import vanilla_roll.array_api as xp
from vanilla_roll.backend import ArrayApiBackend, get_array_api_backend

from .type import Number, SamplingMethod

if TYPE_CHECKING:
    from .numpy import asnumpy, clip, put, take
elif get_array_api_backend() == ArrayApiBackend.NUMPY:
    from .numpy import asnumpy, clip, put, take
elif get_array_api_backend() == ArrayApiBackend.PYTORCH:
    from .pytorch import asnumpy, clip, put, take
elif get_array_api_backend() == ArrayApiBackend.CUPY:
    from .cupy import asnumpy, clip, put, take
else:
    raise OSError("No array API backend found")


def diag(array: xp.Array) -> xp.Array:
    if array.ndim != 1:
        raise ValueError("Expected 1D array")

    n = array.shape[0]
    ret = xp.zeros((n, n), dtype=array.dtype)
    for i in range(n):
        ret[i, i] = array[i]
    return ret


def ravel_index(indices: xp.Array, /, shape: Sequence[int]) -> xp.Array:
    steps = [1]
    for s in reversed(shape):
        steps.append(steps[-1] * s)
    steps_array = xp.asarray(list(reversed(steps[:-1])))
    return indices.T @ steps_array


def assign(array: xp.Array, /, *, indices: xp.Array, values: xp.Array) -> xp.Array:
    new_array = xp.asarray(array, copy=True)
    put(new_array, indices=indices, values=values)
    return new_array


def sample_nearest(array: xp.Array, /, *, coordinates: xp.Array) -> xp.Array:
    coordinates = clip(
        coordinates, a_min=xp.zeros(3), a_max=xp.asarray(array.shape) - 1
    )
    coordinates = xp.astype(xp.reshape(coordinates, (-1, 3)), xp.int64)
    indices = ravel_index(coordinates.T, array.shape)
    return take(array, indices=indices)


def sample_linear(array: xp.Array, /, *, coordinates: xp.Array) -> xp.Array:
    def _create_mask(array: xp.Array, coordinates: xp.Array) -> xp.Array:
        mask = xp.logical_and(0 <= coordinates, coordinates < xp.asarray(array.shape))

        ret = xp.logical_and(mask[:, 0], mask[:, 1])
        if array.ndim == 3:
            ret = xp.logical_and(ret, mask[:, 2])
        return ret

    ret = xp.zeros(coordinates.shape[0])
    org = xp.astype(xp.floor(coordinates), xp.int32)
    diff = coordinates - xp.astype(org, xp.float64)
    shape = array.shape

    for s in itertools.product(*([[0, 1]] * org.shape[-1])):
        step = xp.asarray(s)
        cur = org + step

        mask = _create_mask(array, cur)

        cur_indices = ravel_index(cur.T, shape)
        step = xp.astype(step, xp.float64)
        cur_weight = xp.prod((1 - step) - (1 - 2 * step) * diff, axis=-1)

        indices = cur_indices[mask]
        weight = cur_weight[mask]

        values = take(array, indices=indices)
        ret[mask] += weight * xp.astype(values, xp.float64)
    return ret


def sample(
    array: xp.Array,
    /,
    *,
    coordinates: xp.Array,
    method: type.SamplingMethod = "linear",
) -> xp.Array:
    match method:
        case "linear":
            return sample_linear(array, coordinates=coordinates)
        case "nearest":
            return sample_nearest(array, coordinates=coordinates)


__all__ = [
    "asnumpy",
    "take",
    "put",
    "assign",
    "clip",
    "sample",
    "ravel_index",
    "diag",
    "Number",
    "SamplingMethod",
]

# type: ignore

import itertools
from typing import Any, Literal, Sequence

import numpy.typing as npt

import vanilla_roll.array_api as xp
from vanilla_roll.backend import ArrayApiBackend, get_array_api_backend

Number = float | int | bool
SamplingMethod = Literal["linear"] | Literal["nearest"]


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
    method: SamplingMethod = "linear",
) -> xp.Array:
    match method:
        case "linear":
            return sample_linear(array, coordinates=coordinates)
        case "nearest":
            return sample_nearest(array, coordinates=coordinates)


def take(array: xp.Array, /, *, indices: xp.Array) -> xp.Array:
    ...


def put(array: xp.Array, /, *, indices: xp.Array, values: xp.Array) -> None:
    ...


def clip(
    array: xp.Array,
    /,
    *,
    a_min: xp.Array | Number | None,
    a_max: xp.Array | Number | None = None,
) -> xp.Array:
    ...


def asnumpy(array: xp.Array) -> npt.NDArray[Any]:
    ...


if get_array_api_backend() == ArrayApiBackend.NUMPY:

    def _np_get_raw_array(array: xp.Array) -> npt.NDArray[Any]:
        raw_array = getattr(array, "_array", None)
        if raw_array is None:
            raise ValueError(f"Expected numpy array, got {type(array)}")
        return raw_array

    def _np_take(array: xp.Array, /, *, indices: xp.Array) -> xp.Array:
        raw_array = _np_get_raw_array(array)
        raw_indices = _np_get_raw_array(indices)
        return xp.asarray(raw_array.reshape(-1)[raw_indices])

    def _np_put(array: xp.Array, /, *, indices: xp.Array, values: xp.Array) -> None:
        raw_array = _np_get_raw_array(array)
        raw_indices = _np_get_raw_array(indices)
        raw_values = _np_get_raw_array(values)
        raw_array.reshape(-1)[raw_indices] = raw_values

    def _np_clip(
        array: xp.Array,
        /,
        *,
        a_min: xp.Array | Number | None,
        a_max: xp.Array | Number | None = None,
    ) -> xp.Array:
        raw_array = _np_get_raw_array(array)
        raw_a_min = _np_get_raw_array(a_min) if isinstance(a_min, xp.Array) else a_min
        raw_a_max = _np_get_raw_array(a_max) if isinstance(a_max, xp.Array) else a_max
        return xp.asarray(raw_array.clip(raw_a_min, raw_a_max))  # type: ignore

    def _np_asnumpy(array: xp.Array) -> npt.NDArray[Any]:
        return _np_get_raw_array(array)

    locals().update(
        {"take": _np_take, "put": _np_put, "clip": _np_clip, "asnumpy": _np_asnumpy}
    )

elif get_array_api_backend() == ArrayApiBackend.PYTORCH:
    from typing import cast

    import torch

    def _torch_take(array: xp.Array, /, *, indices: xp.Array) -> xp.Array:
        raw_array = cast(torch.Tensor, array)
        raw_indices = cast(torch.Tensor, indices)
        return xp.asarray(torch.take(raw_array, raw_indices))

    def _torch_put(array: xp.Array, /, *, indices: xp.Array, values: xp.Array) -> None:
        raw_array = cast(torch.Tensor, array)
        raw_indices = cast(torch.Tensor, indices)
        raw_values = cast(torch.Tensor, values)
        raw_array.put_(raw_indices, raw_values)

    def _torch_clip(
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

    def _torch_asnumpy(array: xp.Array) -> npt.NDArray[Any]:
        return array.to("cpu").detach().numpy()

    locals().update(
        {
            "take": _torch_take,
            "put": _torch_put,
            "clip": _torch_clip,
            "asnumpy": _torch_asnumpy,
        }
    )

elif get_array_api_backend() == ArrayApiBackend.CUPY:
    import cupy
    import cupy.typing as cpt

    def _cupy_get_raw_array(array: xp.Array) -> cpt.NDArray[Any]:
        raw_array = getattr(array, "_array", None)
        if raw_array is None:
            raise ValueError(f"Expected numpy array, got {type(array)}")
        return raw_array

    def _cupy_take(array: xp.Array, /, *, indices: xp.Array) -> xp.Array:
        raw_array = _cupy_get_raw_array(array)
        raw_indices = _cupy_get_raw_array(indices)
        return xp.asarray(raw_array.reshape(-1)[raw_indices])

    def _cupy_put(array: xp.Array, /, *, indices: xp.Array, values: xp.Array) -> None:
        raw_array = _cupy_get_raw_array(array)
        raw_indices = _cupy_get_raw_array(indices)
        raw_values = _cupy_get_raw_array(values)
        raw_array.reshape(-1)[raw_indices] = raw_values

    def _cupy_clip(
        array: xp.Array,
        /,
        *,
        a_min: xp.Array | Number | None,
        a_max: xp.Array | Number | None = None,
    ) -> xp.Array:
        raw_array = _cupy_get_raw_array(array)
        raw_a_min = _cupy_get_raw_array(a_min) if isinstance(a_min, xp.Array) else a_min
        raw_a_max = _cupy_get_raw_array(a_max) if isinstance(a_max, xp.Array) else a_max
        return xp.asarray(raw_array.clip(raw_a_min, raw_a_max))  # type: ignore

    def _cupy_asnumpy(array: xp.Array) -> npt.NDArray[Any]:
        return cupy.asnumpy(array._array)

    locals().update(
        {
            "take": _cupy_take,
            "put": _cupy_put,
            "clip": _cupy_clip,
            "asnumpy": _cupy_asnumpy,
        }
    )
else:
    raise OSError("No array API backend found")

__all__ = [
    "take",
    "put",
    "assign",
    "clip",
    "sample",
    "ravel_index",
    "diag",
]

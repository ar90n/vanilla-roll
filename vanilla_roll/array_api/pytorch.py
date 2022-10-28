# pyright: reportUnusedImport=false
# pyright: reportUnknownVariableType=false
# pyright: reportWildcardImportFromLibrary=false

import builtins

import torch.linalg as linalg  # noqa: F401
from torch import *  # noqa: F403
from torch import permute  # noqa: F401
from torch import uint8  # noqa: F401
from torch import argsort as _argsort
from torch import bool, dot, float32, float64, int8, int16, int32, int64  # noqa: F401
from torch import max as _max
from torch import min as _min
from torch import sort as _sort
from torch.utils.dlpack import from_dlpack  # noqa: F401

Array = Tensor  # noqa: F405


def max(
    array: Array,
    /,
    *,
    axis: builtins.int | None = None,
    keepdims: builtins.bool = False,
) -> Array:
    if axis is None:
        ret = _max(array)
        return ret

    return _max(array, axis, False)[0]


def min(
    array: Array,
    /,
    *,
    axis: builtins.int | None = None,
    keepdims: builtins.bool = False,
) -> Array:
    if axis is None:
        ret = _min(array)
        return ret

    return _min(array, axis, False)[0]


def permute_dims(array: Array, /, axes: tuple[builtins.int, ...]) -> Array:
    return permute(array, axes)


def astype(array: Array, /, _dtype: dtype) -> Array:  # noqa: F405
    return array.to(_dtype)


def vecdot(arr1: Array, arr2: Array, /, *, axis: builtins.int = -1) -> Array:
    return dot(arr1, arr2)


def argsort(
    array: Array,
    /,
    *,
    axis: builtins.int = -1,
    descending: builtins.bool = False,
    stable: builtins.bool = True,
) -> Array:
    return _argsort(array, axis, descending)


def sort(
    array: Array,
    /,
    *,
    axis: builtins.int = -1,
    descending: builtins.bool = False,
    stable: builtins.bool = True,
) -> Array:
    return _sort(array, axis, descending)


#

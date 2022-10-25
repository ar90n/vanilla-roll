# pyright: reportWildcardImportFromLibrary=false
from typing import TYPE_CHECKING

from vanilla_roll.backend import ArrayApiBackend, get_array_api_backend

if TYPE_CHECKING:
    from numpy.array_api import *  # noqa: F403
    from numpy.array_api._array_object import Array  # pyright: reportUnusedImport=false
elif get_array_api_backend() == ArrayApiBackend.NUMPY:
    from numpy.array_api import *  # noqa: F403
    from numpy.array_api._array_object import Array  # pyright: reportUnusedImport=false
elif get_array_api_backend() == ArrayApiBackend.PYTORCH:
    import torch
    import torch.linalg as linalg  # noqa: F401
    from torch import *  # noqa: F403
    from torch import (  # noqa: F401
        bool,
        float32,
        float64,
        from_dlpack,
        int8,
        int16,
        int32,
        int64,
        uint8,
    )

    Array = Tensor  # noqa: F405

    def max(
        array: Array, /, *, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        return torch.max(array, axis=axis, keepdims=keepdims)[0]

    def min(
        array: Array, /, *, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        return torch.min(array, axis=axis, keepdims=keepdims)[0]

    def permute_dims(array: Array, /, axes: tuple[int, ...]) -> Array:
        return torch.permute(array, axes)

    def astype(array: Array, /, _dtype: dtype) -> Array:  # noqa: F405
        return array.to(_dtype)

elif get_array_api_backend() == ArrayApiBackend.CUPY:
    # from cupy.array_api import *  # noqa: F401, F403
    from cupy.array_api import *  # noqa
    from cupy.array_api._array_object import Array  # pyright: reportUnusedImport=false
else:
    raise OSError("No array API backend found")

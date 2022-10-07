import os
from enum import Enum

from vanilla_roll.util import has_module


class ArrayApiBackend(Enum):
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    CUPY = "cupy"


def has_backend(backend_name: str) -> bool:
    return {"numpy": HAS_NUMPY, "pytorch": HAS_PYTORCH, "cupy": HAS_CUPY}.get(
        backend_name, False
    )


HAS_NUMPY = has_module("numpy")
HAS_PYTORCH = has_module("torch")
HAS_CUPY = has_module("cupy")


def get_array_api_backend() -> ArrayApiBackend | None:
    match ArrayApiBackend(os.environ.get("ARRAY_API_BACKEND", "numpy")):
        case ArrayApiBackend.NUMPY if HAS_NUMPY:
            return ArrayApiBackend.NUMPY
        case ArrayApiBackend.PYTORCH if HAS_PYTORCH:
            return ArrayApiBackend.PYTORCH
        case ArrayApiBackend.CUPY if HAS_CUPY:
            return ArrayApiBackend.CUPY
        case _:
            return None

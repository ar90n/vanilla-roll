import os
from enum import Enum

from vanilla_roll.util import has_module


class ArrayApiBackend(Enum):
    NUMPY = "numpy"
    PYTORCH = "pytorch"


def has_backend(backend_name: str) -> bool:
    return {"numpy": HAS_NUMPY, "pytorch": HAS_PYTORCH}.get(backend_name, False)


HAS_NUMPY = has_module("numpy")
HAS_PYTORCH = has_module("torch")


def get_array_api_backend() -> ArrayApiBackend | None:
    match ArrayApiBackend(os.environ.get("ARRAY_API_BACKEND", "numpy")):
        case ArrayApiBackend.NUMPY if HAS_NUMPY:
            return ArrayApiBackend.NUMPY
        case ArrayApiBackend.PYTORCH if HAS_PYTORCH:
            return ArrayApiBackend.PYTORCH
        case _:
            return None

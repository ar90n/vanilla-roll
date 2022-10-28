from typing import TYPE_CHECKING

from vanilla_roll.backend import ArrayApiBackend, get_array_api_backend

if TYPE_CHECKING:
    from .numpy import *  # noqa: F403
elif get_array_api_backend() == ArrayApiBackend.NUMPY:
    from .numpy import *  # noqa: F403, F401
elif get_array_api_backend() == ArrayApiBackend.PYTORCH:
    from .pytorch import *  # noqa: F403, F401
elif get_array_api_backend() == ArrayApiBackend.CUPY:
    from .cupy import *  # noqa: F403, F401
else:
    raise OSError("No array API backend found")

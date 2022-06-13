import os
from importlib import reload
from unittest import mock

import pytest

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
import vanilla_roll.backend as backend


@pytest.fixture(params=["numpy", "pytorch"])
def array_api_backend(request: pytest.FixtureRequest):
    backend_name = str(request.param)  # type: ignore

    if not backend.has_backend(backend_name):
        pytest.skip("No backend found")

    with mock.patch.dict(os.environ, {"ARRAY_API_BACKEND": backend_name}):
        reload(xp)
        reload(xpe)
        yield


class Helpers:
    @staticmethod
    def approx_equal(lhs: xp.Array, rhs: xp.Array) -> bool:
        return bool(xp.all(xp.abs(lhs - rhs) < 1e-6))


@pytest.fixture
def helpers():
    return Helpers

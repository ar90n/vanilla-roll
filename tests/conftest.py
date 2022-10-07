import os
from importlib import reload
from unittest import mock

import pytest

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
import vanilla_roll.backend as backend
from vanilla_roll.anatomy_orientation import (
    CSA,
    AnatomyOrientation,
    Axial,
    Coronal,
    Sagittal,
)
from vanilla_roll.geometry.element import Frame, Orientation, Vector, world_frame
from vanilla_roll.volume import Volume


@pytest.fixture(params=["numpy", "pytorch", "cupy"])
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

    @staticmethod
    def create_volume(
        *,
        data: xp.Array | None = None,
        shape: tuple[int, int, int] = (256, 256, 256),
        origin: Vector = world_frame.origin,
        orientation: Orientation = world_frame.orientation,
        anatomy_orientation: AnatomyOrientation = CSA(
            Coronal.LEFT, Sagittal.ANTERIOR, Axial.SUPERIOR
        ),
    ) -> Volume:
        if data is None:
            data = xp.zeros(shape, dtype=xp.float64)

        frame = Frame(origin=origin, orientation=orientation)

        return Volume(data=data, frame=frame, anatomy_orientation=anatomy_orientation)


@pytest.fixture
def helpers():
    return Helpers

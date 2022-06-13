import os
from importlib import reload
from typing import Any
from unittest import mock

import pytest

import vanilla_roll.array_api as xp
from vanilla_roll import backend


@pytest.mark.parametrize(
    "backend_name, should_skip, expected",
    [
        ("numpy", not backend.HAS_NUMPY, "numpy.array_api._array_object.Array"),
        ("pytorch", not backend.HAS_PYTORCH, "torch.Tensor"),
    ],
)
def test_import(backend_name: str, should_skip: bool, expected: str):
    if should_skip:
        pytest.skip("No backend found")

    with mock.patch.dict(os.environ, {"ARRAY_API_BACKEND": backend_name}):
        reload(xp)

        assert f"{xp.Array.__module__}.{xp.Array.__name__}" == expected
        assert xp.asarray([1, 2, 3]) is not None


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "lhs, rhs, op, expected",
    [
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], "==", [True, True, True]),
        ([0.0, 1.0, 2.0], [3.0, 4.0, 5.0], "+", [3.0, 5.0, 7.0]),
    ],
)
def test_ops(
    lhs: list[float],
    rhs: list[float],
    op: str,
    expected: list[Any],
):
    lhs_array = xp.asarray(lhs)  # type: ignore # noqa: F841
    rhs_array = xp.asarray(rhs)  # type: ignore # noqa: F841
    expected_array = xp.asarray(expected)
    actual_array = eval(f"lhs_array {op} rhs_array")
    assert xp.all(actual_array == expected_array)

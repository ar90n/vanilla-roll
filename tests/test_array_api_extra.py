from typing import Any

import pytest

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "array, indices, expected",
    [
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0, 1, 5], [0.0, 1.0, 5.0]),
        ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [0, 1, 5], [0.0, 1.0, 5.0]),
    ],
)
def test_take(
    array: list[float],
    indices: list[int],
    expected: list[float],
):
    actual_array = xpe.take(xp.asarray(array), indices=xp.asarray(indices))
    expected_array = xp.asarray(expected)
    assert xp.all(actual_array == expected_array)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "array, indices, values, expected",
    [
        (
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0, 1, 5],
            [-1.0, -2.0, -3.0],
            [-1.0, -2.0, 2.0, 3.0, 4.0, -3.0],
        ),
        (
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [0, 1, 5],
            [-1.0, -2.0, -3.0],
            [[-1.0, -2.0, 2.0], [3.0, 4.0, -3.0]],
        ),
    ],
)
def test_put(
    array: list[float],
    indices: list[int],
    values: list[float],
    expected: list[float],
):
    actual_array = xp.asarray(array)
    expected_array = xp.asarray(expected)
    xpe.put(actual_array, indices=xp.asarray(indices), values=xp.asarray(values))
    assert xp.all(actual_array == expected_array)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "array, indices, values, expected",
    [
        (
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0, 1, 5],
            [-1.0, -2.0, -3.0],
            [-1.0, -2.0, 2.0, 3.0, 4.0, -3.0],
        ),
        (
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [0, 1, 5],
            [-1.0, -2.0, -3.0],
            [[-1.0, -2.0, 2.0], [3.0, 4.0, -3.0]],
        ),
    ],
)
def test_assign(
    array: list[float],
    indices: list[int],
    values: list[float],
    expected: list[float],
):
    actual_array = xpe.assign(
        xp.asarray(array), indices=xp.asarray(indices), values=xp.asarray(values)
    )
    expected_array = xp.asarray(expected)
    assert xp.all(actual_array == expected_array)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "array, a_min, a_max, expected",
    [
        (
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            0.0,
            None,
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        ),
        (
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            None,
            0.0,
            [-2.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            -1.0,
            2.0,
            [-1.0, -1.0, 0.0, 1.0, 2.0, 2.0],
        ),
        (
            [[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]],
            -1.0,
            2.0,
            [[-1.0, -1.0, 0.0], [1.0, 2.0, 2.0]],
        ),
        (
            [
                [-3.0, -3.0],
                [-2.0, -2.0],
                [-1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            [-1.0, -2.0],
            [2.0, 1.0],
            [
                [-1.0, -2.0],
                [-1.0, -2.0],
                [-1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 1.0],
            ],
        ),
    ],
)
def test_clip(
    array: list[float],
    a_min: list[float] | float | None,
    a_max: list[float] | float | None,
    expected: list[float],
):
    actual_a_min = xp.asarray(a_min) if isinstance(a_min, list) else a_min
    actual_a_max = xp.asarray(a_max) if isinstance(a_max, list) else a_max
    actual_array = xpe.clip(xp.asarray(array), a_min=actual_a_min, a_max=actual_a_max)
    expected_array = xp.asarray(expected)
    assert xp.all(actual_array == expected_array)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "indices, shape, expected",
    [
        ([[0, 1, 2], [0, 1, 3]], [3, 4], [0, 5, 11]),
        ([[3, 7, 2], [7, 0, 3]], [8, 8], [31, 56, 19]),
        ([[1, 0, 1], [2, 1, 0], [3, 1, 1]], [2, 3, 4], [23, 5, 13]),
    ],
)
def test_ravel_index(
    indices: list[list[int]],
    shape: list[int],
    expected: list[int],
):
    actual_indices = xpe.ravel_index(xp.asarray(indices), shape=shape)
    expected_indices = xp.asarray(expected)
    assert xp.all(actual_indices == expected_indices)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "array, coords, expected",
    [
        ([[0, 1], [1, 2]], [[0.5, 0.5]], [1.0]),
        (
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[0.1, 0.5], [1.2, 0.5], [1.3, 1.8]],
            [0.8, 4.1, 5.7],
        ),
        (
            [
                [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
            ],
            [[0.1, 0.5, 1.2], [1.2, 0.5, 1.8], [0.0, 2.0, 1.8]],
            [3.6, 14.1, 7.8],
        ),
    ],
)
def test_sample(
    array: list[list[int]],
    coords: list[list[float]],
    expected: list[float],
    helpers: Any,
):
    actual_array = xp.asarray(array)
    actual_coords = xp.asarray(coords)
    actual_sampled = xpe.sample(actual_array, coordinates=actual_coords)
    expected_array = xp.asarray(expected)
    assert helpers.approx_equal(actual_sampled, expected_array)


@pytest.mark.usefixtures("array_api_backend")
@pytest.mark.parametrize(
    "values, expected",
    [
        ([8, 1, 2], [[8, 0, 0], [0, 1, 0], [0, 0, 2]]),
        ([8.0, 1.0, 2.0], [[8.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
    ],
)
def test_diag(
    values: list[int | float],
    expected: list[list[int | float]],
):
    actual_array = xpe.diag(xp.asarray(values))
    expected_array = xp.asarray(expected)
    assert xp.all(actual_array == expected_array)

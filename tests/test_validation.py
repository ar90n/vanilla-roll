import pytest

from vanilla_roll.validation import (
    IsFinite,
    IsGreaterEqualThan,
    IsGreaterThan,
    IsLessEqualThan,
    IsLessThan,
    ValidationRule,
    Validator,
)


@pytest.mark.parametrize(
    "value, rules",
    [
        (2.0, [IsGreaterThan(1.0)]),
        (float("nan"), [IsGreaterThan(1.0)]),
    ],
)
def test_is_greater_than(value: float, rules: list[ValidationRule]):
    assert Validator(rules)("value", value) is None


@pytest.mark.parametrize(
    "value, rules",
    [
        (1.0, [IsGreaterThan(1.0)]),
        (0.0, [IsGreaterThan(1.0)]),
    ],
)
def test_falilure_is_greater_than(value: float, rules: list[ValidationRule]):
    with pytest.raises(ValueError):
        Validator(rules)("value", value)


@pytest.mark.parametrize(
    "value, rules",
    [
        (2.0, [IsGreaterEqualThan(1.0)]),
        (1.0, [IsGreaterEqualThan(1.0)]),
        (float("nan"), [IsGreaterThan(1.0)]),
    ],
)
def test_is_greater_equal_than(value: float, rules: list[ValidationRule]):
    assert Validator(rules)("value", value) is None


@pytest.mark.parametrize(
    "value, rules",
    [
        (0.0, [IsGreaterEqualThan(1.0)]),
    ],
)
def test_falilure_is_greater_equal_than(value: float, rules: list[ValidationRule]):
    with pytest.raises(ValueError):
        Validator(rules)("value", value)


@pytest.mark.parametrize(
    "value, rules",
    [
        (0.0, [IsLessThan(1.0)]),
        (-float("nan"), [IsGreaterThan(1.0)]),
    ],
)
def test_is_less_than(value: float, rules: list[ValidationRule]):
    assert Validator(rules)("value", value) is None


@pytest.mark.parametrize(
    "value, rules",
    [
        (1.0, [IsLessThan(1.0)]),
        (2.0, [IsLessThan(1.0)]),
    ],
)
def test_falilure_is_less_than(value: float, rules: list[ValidationRule]):
    with pytest.raises(ValueError):
        Validator(rules)("value", value)


@pytest.mark.parametrize(
    "value, rules",
    [
        (1.0, [IsLessEqualThan(1.0)]),
        (0.0, [IsLessEqualThan(1.0)]),
        (-float("nan"), [IsGreaterThan(1.0)]),
    ],
)
def test_is_less_equal_than(value: float, rules: list[ValidationRule]):
    assert Validator(rules)("value", value) is None


@pytest.mark.parametrize(
    "value, rules",
    [
        (2.0, [IsLessEqualThan(1.0)]),
    ],
)
def test_falilure_is_less_equal_than(value: float, rules: list[ValidationRule]):
    with pytest.raises(ValueError):
        Validator(rules)("value", value)


@pytest.mark.parametrize(
    "value, rules",
    [
        (-1.0, [IsFinite()]),
        (0.0, [IsFinite()]),
        (1.0, [IsFinite()]),
    ],
)
def test_is_finite(value: float, rules: list[ValidationRule]):
    assert Validator(rules)("value", value) is None


@pytest.mark.parametrize(
    "value, rules",
    [
        (float("nan"), [IsFinite()]),
        (float("inf"), [IsFinite()]),
        (-float("inf"), [IsFinite()]),
    ],
)
def test_falilure_is_finite(value: float, rules: list[ValidationRule]):
    with pytest.raises(ValueError):
        Validator(rules)("value", value)


@pytest.mark.parametrize(
    "value, rules",
    [
        (1.0, [IsGreaterThan(0.0), IsLessThan(2.0)]),
        (-8.0, [IsFinite(), IsLessThan(2.0)]),
    ],
)
def test_complex_rule(value: float, rules: list[ValidationRule]):
    assert Validator(rules)("value", value) is None


@pytest.mark.parametrize(
    "value, rules",
    [
        (-1.0, [IsGreaterThan(0.0), IsLessThan(2.0)]),
        (3.0, [IsGreaterThan(0.0), IsLessThan(2.0)]),
        (8.0, [IsFinite(), IsLessThan(2.0)]),
        (-float("inf"), [IsFinite(), IsLessThan(2.0)]),
    ],
)
def test_falilure_complex_rule(value: float, rules: list[ValidationRule]):
    with pytest.raises(ValueError):
        Validator(rules)("value", value)

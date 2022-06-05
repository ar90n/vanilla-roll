import math
from typing import Any, Iterable, Protocol


class ValidationRule(Protocol):
    """ValidationRule is a protocol that defines a validation rule."""

    def __call__(self, name: str, other: Any) -> ValueError | None:
        ...


class IsGreaterThan:
    _value: float

    def __init__(self, value: float):
        self._value = value

    def __call__(self, name: str, other: float) -> ValueError | None:
        if other <= self._value:
            return ValueError(f"{name} must be greater than {self._value}. Got {other}")


class IsGreaterEqualThan:
    _value: float

    def __init__(self, value: float):
        self._value = value

    def __call__(self, name: str, other: float) -> ValueError | None:
        if other < self._value:
            return ValueError(
                f"{name} must be greater equal than {self._value}. Got {other}"
            )


class IsLessThan:
    _value: float

    def __init__(self, value: float):
        self._value = value

    def __call__(self, name: str, other: float) -> ValueError | None:
        if other >= self._value:
            return ValueError(f"{name} must be less than {self._value}. Got {other}")


class IsLessEqualThan:
    _value: float

    def __init__(self, value: float):
        self._value = value

    def __call__(self, name: str, other: float) -> ValueError | None:
        if other > self._value:
            return ValueError(
                f"{name} must be less equal than {self._value}. Got {other}"
            )


class IsFinite:
    def __call__(self, name: str, other: float) -> ValueError | None:
        if math.isnan(other) or math.isinf(other):
            return ValueError(f"{name} must be finite. Got {other}")


class Validator:
    _rules: list[ValidationRule]

    def __init__(self, rules: Iterable[ValidationRule]):
        self._rules = [*rules]

    def __call__(self, name: str, value: Any) -> Exception | None:
        for rule in self._rules:
            if exception := rule(name, value):
                raise exception

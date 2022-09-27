from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Generic, Literal, TypeAlias, TypeVar, get_args

from vanilla_roll.geometry.element import Vector


class Axial(Enum):
    SUPERIOR = "Superior"
    INFERIOR = "Inferior"

    @classmethod
    def of(cls, s: Literal["Superior"] | Literal["Inferior"]) -> Axial:
        """construct from string.

        >>> Axial.of("Superior")
        <Axial.SUPERIOR: 'Superior'>
        """
        return cls(s)

    def inverse(self) -> Axial:
        """get inverse.

        >>> Axial.SUPERIOR.inverse()
        <Axial.INFERIOR: 'Inferior'>
        """
        return Axial.SUPERIOR if self == Axial.INFERIOR else Axial.INFERIOR


class Sagittal(Enum):
    ANTERIOR = "Anterior"
    POSTERIOR = "Posterior"

    @classmethod
    def of(cls, s: Literal["Anterior"] | Literal["Posterior"]) -> Sagittal:
        """construct from string.

        >>> Sagittal.of("Anterior")
        <Sagittal.ANTERIOR: 'Anterior'>
        """
        return cls(s)

    def inverse(self) -> Sagittal:
        """
        >>> Sagittal.ANTERIOR.inverse()
        <Sagittal.POSTERIOR: 'Posterior'>
        """
        return Sagittal.ANTERIOR if self == Sagittal.POSTERIOR else Sagittal.POSTERIOR


class Coronal(Enum):
    RIGHT = "Right"
    LEFT = "Left"

    @classmethod
    def of(cls, s: Literal["Right"] | Literal["Left"]) -> Coronal:
        """construct from string.

        >>> Coronal.of("Right")
        <Coronal.RIGHT: 'Right'>
        """
        return cls(s)

    def inverse(self) -> Coronal:
        """
        >>> Coronal.RIGHT.inverse()
        <Coronal.LEFT: 'Left'>
        """
        return Coronal.LEFT if self == Coronal.RIGHT else Coronal.RIGHT


AnatomyAxis = Axial | Sagittal | Coronal

T = TypeVar("T", bound=AnatomyAxis)
I = TypeVar("I", bound=AnatomyAxis)  # noqa: E741
J = TypeVar("J", bound=AnatomyAxis)
K = TypeVar("K", bound=AnatomyAxis)


@dataclass(frozen=True)
class _AnatomyOrientation(Generic[I, J, K]):
    i: I
    j: J
    k: K


ASC: TypeAlias = _AnatomyOrientation[Axial, Sagittal, Coronal]
ACS: TypeAlias = _AnatomyOrientation[Axial, Coronal, Sagittal]
SAC: TypeAlias = _AnatomyOrientation[Sagittal, Axial, Coronal]
SCA: TypeAlias = _AnatomyOrientation[Sagittal, Coronal, Axial]
CAS: TypeAlias = _AnatomyOrientation[Coronal, Axial, Sagittal]
CSA: TypeAlias = _AnatomyOrientation[Coronal, Sagittal, Axial]

AnatomyOrientation = ASC | ACS | SAC | SCA | CAS | CSA


def create(
    i: AnatomyAxis,
    j: AnatomyAxis,
    k: AnatomyAxis,
) -> AnatomyOrientation:
    """
    >>> create(Axial.SUPERIOR, Sagittal.ANTERIOR, Coronal.RIGHT)
    _AnatomyOrientation(i=<Axial.SUPERIOR: 'Superior'>, j=<Sagittal.ANTERIOR: 'Anterior'>, k=<Coronal.RIGHT: 'Right'>)
    >>> create(Axial.SUPERIOR, Sagittal.ANTERIOR, Axial.SUPERIOR)
    Traceback (most recent call last):
    ...
    ValueError: Invalid anatomy orientation: Axial.SUPERIOR-Sagittal.ANTERIOR-Axial.SUPERIOR
    """
    for cls in (ASC, ACS, SAC, SCA, CAS, CSA):
        c0, c1, c2 = get_args(cls)
        if isinstance(i, c0) and isinstance(j, c1) and isinstance(k, c2):
            # the types of i, j and k are valid combinations for anatomy orientation
            return cls(i=i, j=j, k=k)  # type: ignore

    raise ValueError(f"Invalid anatomy orientation: {i}-{j}-{k}")


def parse(s: str) -> AnatomyOrientation:
    """
    >>> parse("Superior-Anterior-Right")
    _AnatomyOrientation(i=<Axial.SUPERIOR: 'Superior'>, j=<Sagittal.ANTERIOR: 'Anterior'>, k=<Coronal.RIGHT: 'Right'>)
    >>> parse("ALI")
    _AnatomyOrientation(i=<Sagittal.ANTERIOR: 'Anterior'>, j=<Coronal.LEFT: 'Left'>, k=<Axial.INFERIOR: 'Inferior'>)
    >>> parse("AAA")
    Traceback (most recent call last):
    ...
    ValueError: Invalid anatomy orientation: AAA
    """

    anatomy_dirs = _normalize_orientation_if_possible(s).split("-")
    if len(anatomy_dirs) != 3:
        raise ValueError(f"Invalid anatomy orientation: {s}")

    for cls in (ASC, ACS, SAC, SCA, CAS, CSA):
        c0, c1, c2 = get_args(cls)

        if (
            (i := _try_parse_direction(c0, anatomy_dirs[0]))
            and (j := _try_parse_direction(c1, anatomy_dirs[1]))
            and (k := _try_parse_direction(c2, anatomy_dirs[2]))
        ):
            return cls(i, j, k)

    raise ValueError(f"Invalid anatomy orientation: {s}")


def get_direction(orientation: AnatomyOrientation, axis: AnatomyAxis) -> Vector:
    """
    >>> get_direction(parse("SAR"), Axial.SUPERIOR)
    Vector(i=1.0, j=0.0, k=0.0)
    """

    return {
        orientation.i: Vector(i=1.0, j=0.0, k=0.0),
        orientation.i.inverse(): Vector(i=-1.0, j=0.0, k=0.0),
        orientation.j: Vector(i=0.0, j=1.0, k=0.0),
        orientation.j.inverse(): Vector(i=0.0, j=-1.0, k=0.0),
        orientation.k: Vector(i=0.0, j=0.0, k=1.0),
        orientation.k.inverse(): Vector(i=0.0, j=0.0, k=-1.0),
    }[axis]


@lru_cache(maxsize=None)
def _get_fullname_mapping(cls: type[Enum]) -> dict[str, str]:
    return {
        **{k[0].upper(): k for k in cls._value2member_map_.keys()},
        **{k.upper(): k for k in cls._value2member_map_.keys()},
    }


def _is_enum(cls: type[Enum], s: str) -> bool:
    return s in cls._value2member_map_


def _normalize_axis_if_possible(cls: type[Enum], s: str) -> str:
    return _get_fullname_mapping(cls).get(s.upper(), s)


def _try_parse_direction(cls: type[T], s: str) -> T | None:
    s = _normalize_axis_if_possible(cls, s)
    if _is_enum(cls, s):
        return cls(s)
    return None


def _normalize_orientation_if_possible(s: str) -> str:
    candidate_delimiters = [" ", "_", ",", ":"]
    for delimiter in candidate_delimiters:
        s = s.replace(delimiter, "-")

    if len(s) == 3 and s.count("-") == 0:
        return "-".join(s)

    return s

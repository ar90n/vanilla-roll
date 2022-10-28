import pytest

from vanilla_roll.anatomy_orientation import (
    ACS,
    ASC,
    CAS,
    CSA,
    SAC,
    SCA,
    AnatomyAxis,
    AnatomyOrientation,
    Axial,
    Coronal,
    Sagittal,
    create,
    get_direction,
    parse,
)
from vanilla_roll.geometry.element import Vector


def test_axial():
    assert Axial.SUPERIOR == Axial.of("Superior")
    assert Axial.INFERIOR == Axial.of("Inferior")
    assert Axial.SUPERIOR.inverse() == Axial.INFERIOR
    assert Axial.INFERIOR.inverse() == Axial.SUPERIOR


def test_sagittal():
    assert Sagittal.ANTERIOR == Sagittal.of("Anterior")
    assert Sagittal.POSTERIOR == Sagittal.of("Posterior")
    assert Sagittal.ANTERIOR.inverse() == Sagittal.POSTERIOR
    assert Sagittal.POSTERIOR.inverse() == Sagittal.ANTERIOR


def test_coronal():
    assert Coronal.RIGHT == Coronal.of("Right")
    assert Coronal.LEFT == Coronal.of("Left")
    assert Coronal.RIGHT.inverse() == Coronal.LEFT
    assert Coronal.LEFT.inverse() == Coronal.RIGHT


@pytest.mark.parametrize(
    "i, j, k, expected",
    [
        (
            Axial.SUPERIOR,
            Sagittal.ANTERIOR,
            Coronal.RIGHT,
            ASC(i=Axial.SUPERIOR, j=Sagittal.ANTERIOR, k=Coronal.RIGHT),
        ),
        (
            Axial.SUPERIOR,
            Coronal.RIGHT,
            Sagittal.ANTERIOR,
            ACS(i=Axial.SUPERIOR, j=Coronal.RIGHT, k=Sagittal.ANTERIOR),
        ),
        (
            Sagittal.ANTERIOR,
            Axial.SUPERIOR,
            Coronal.RIGHT,
            SAC(i=Sagittal.ANTERIOR, j=Axial.SUPERIOR, k=Coronal.RIGHT),
        ),
        (
            Sagittal.ANTERIOR,
            Coronal.RIGHT,
            Axial.SUPERIOR,
            SCA(i=Sagittal.ANTERIOR, j=Coronal.RIGHT, k=Axial.SUPERIOR),
        ),
        (
            Coronal.RIGHT,
            Axial.SUPERIOR,
            Sagittal.ANTERIOR,
            CAS(i=Coronal.RIGHT, j=Axial.SUPERIOR, k=Sagittal.ANTERIOR),
        ),
        (
            Coronal.RIGHT,
            Sagittal.ANTERIOR,
            Axial.SUPERIOR,
            CSA(i=Coronal.RIGHT, j=Sagittal.ANTERIOR, k=Axial.SUPERIOR),
        ),
    ],
)
def test_create(
    i: AnatomyAxis, j: AnatomyAxis, k: AnatomyAxis, expected: AnatomyOrientation
):
    actual = create(i, j, k)
    assert expected == actual


def test_create_failed():
    with pytest.raises(ValueError):
        create(Axial.SUPERIOR, Sagittal.ANTERIOR, Axial.SUPERIOR)


@pytest.mark.parametrize(
    "s, expected",
    [
        (
            "Superior-Anterior-Right",
            ASC(i=Axial.SUPERIOR, j=Sagittal.ANTERIOR, k=Coronal.RIGHT),
        ),
        ("SAR", ASC(i=Axial.SUPERIOR, j=Sagittal.ANTERIOR, k=Coronal.RIGHT)),
        (
            "Inferior-Posterior-Left",
            ASC(i=Axial.INFERIOR, j=Sagittal.POSTERIOR, k=Coronal.LEFT),
        ),
        ("IPL", ASC(i=Axial.INFERIOR, j=Sagittal.POSTERIOR, k=Coronal.LEFT)),
    ],
)
def test_parse(s: str, expected: AnatomyOrientation):
    actual = parse(s)
    assert expected == actual


@pytest.mark.parametrize("s", ["AAA", "Superior-Anterior-Inferior"])
def test_parse_failed(s: str):
    with pytest.raises(ValueError):
        parse(s)


@pytest.mark.parametrize(
    "orientation, axis, expected",
    [
        (parse("SAR"), Axial.SUPERIOR, Vector(1.0, 0.0, 0.0)),
        (parse("SAR"), Sagittal.POSTERIOR, Vector(0.0, -1.0, 0.0)),
        (parse("SAR"), Coronal.RIGHT, Vector(0.0, 0.0, 1.0)),
        (parse("IPL"), Axial.SUPERIOR, Vector(-1.0, 0.0, 0.0)),
        (parse("IPL"), Sagittal.POSTERIOR, Vector(0.0, 1.0, 0.0)),
        (parse("IPL"), Coronal.RIGHT, Vector(0.0, 0.0, -1.0)),
    ],
)
def test_get_direction(
    orientation: AnatomyOrientation, axis: AnatomyAxis, expected: Vector
):
    assert expected == get_direction(orientation, axis)

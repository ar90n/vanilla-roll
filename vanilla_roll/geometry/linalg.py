from vanilla_roll.geometry.element import Vector


def norm(vector: Vector) -> float:
    return (vector.i**2 + vector.j**2 + vector.k**2) ** 0.5


def normalize_vector(vector: Vector) -> Vector:
    """Normalize a vector.

    >>> normalize_vector(Vector(1.0, 2.0, 3.0)) # doctest: +ELLIPSIS
    Vector(i=0.26726..., j=0.53452..., k=0.80178...)
    """
    length = (vector.i**2 + vector.j**2 + vector.k**2) ** 0.5
    return Vector(
        i=vector.i / length,
        j=vector.j / length,
        k=vector.k / length,
    )

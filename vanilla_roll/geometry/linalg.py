from vanilla_roll.geometry.element import Vector


def norm(vector: Vector) -> float:
    return (vector.i**2 + vector.j**2 + vector.k**2) ** 0.5


def normalize_vector(vector: Vector) -> Vector:
    """Normalize a vector.

    >>> normalize_vector(Vector(1.0, 2.0, 2.0))
    Vector(i=0.17677669529663687, j=0.35355339059327373, k=0.5303300858899106)
    """
    length = (vector.i**2 + vector.j**2 + vector.k**2) ** 0.5
    return Vector(
        i=vector.i / length,
        j=vector.j / length,
        k=vector.k / length,
    )

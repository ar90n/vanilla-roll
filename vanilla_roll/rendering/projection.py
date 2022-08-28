from dataclasses import dataclass


@dataclass(frozen=True)
class Orthogoal:
    pass


@dataclass(frozen=True)
class Perspective:
    pass


Projection = Orthogoal | Perspective

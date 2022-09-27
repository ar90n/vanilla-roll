from dataclasses import dataclass


@dataclass(frozen=True)
class Sampling:
    step: float


@dataclass(frozen=True)
class ShearWarp:
    pass


Algorithm = Sampling | ShearWarp

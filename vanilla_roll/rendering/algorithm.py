from dataclasses import dataclass


@dataclass(frozen=True)
class Raycast:
    step: float


@dataclass(frozen=True)
class ShearWarp:
    pass


Algorithm = Raycast | ShearWarp

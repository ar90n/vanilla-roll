from dataclasses import dataclass


@dataclass(frozen=True)
class MIP:
    pass


@dataclass(frozen=True)
class MinP:
    pass


@dataclass(frozen=True)
class Average:
    pass


@dataclass(frozen=True)
class VR:
    pass


Mode = MIP | MinP | Average | VR

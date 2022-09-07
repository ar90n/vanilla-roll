from dataclasses import dataclass

from vanilla_roll.rendering.transfer_function import TransferFunction


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
    transfer_function: TransferFunction


Mode = MIP | MinP | Average | VR

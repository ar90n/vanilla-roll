from typing import Callable

import vanilla_roll.array_api_extra as xpe
from vanilla_roll.volume import Volume
from vanilla_roll.rendering.projection import Projection, Orthogoal
from vanilla_roll.rendering.mode import Mode, MinP, MIP, Average
from vanilla_roll.rendering.algorithm import Algorithm, ShearWarp, Raycast
from vanilla_roll.rendering.types import Renderer
from vanilla_roll.rendering.accumulation import Accumulator, AccMax, AccMin, AccMean
from vanilla_roll.rendering.orthogonal_raycast import (
    create_renderer as create_orthogonal_raycast,
)
from vanilla_roll.rendering.orthogonal_shear_warp import (
    create_renderer as create_orthogonal_shear_warp,
)


def _get_accumulator_constructor(
    rendering_mode: Mode,
    sampling_method: xpe.SamplingMethod = "linear",
) -> Callable[[tuple[int, int]], Accumulator]:
    match rendering_mode:
        case MinP():
            return lambda shape: AccMin(shape, sampling_method=sampling_method)
        case MIP():
            return lambda shape: AccMax(shape, sampling_method=sampling_method)
        case Average():
            return lambda shape: AccMean(shape, sampling_method=sampling_method)
        case _:
            raise NotImplementedError(f"{rendering_mode}")


def creaet_renderer(
    volume: Volume,
    projection: Projection,
    rendering_method: Mode,
    sampling_method: xpe.SamplingMethod = "linear",
    algorithm: Algorithm = ShearWarp(),
) -> Renderer:
    accumulator_constructor = _get_accumulator_constructor(rendering_method)
    match (projection, algorithm):
        case (Orthogoal(), Raycast(step)):
            return create_orthogonal_raycast(
                volume,
                step=step,
                accumulator_constructor=accumulator_constructor,
                sampling_method=sampling_method,
            )
        case (Orthogoal(), ShearWarp()):
            return create_orthogonal_shear_warp(
                volume,
                accumulator_constructor=accumulator_constructor,
                sampling_method=sampling_method,
            )
        case _:
            raise NotImplementedError(f"{projection}")

from typing import Callable

import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
from vanilla_roll.rendering.algorithm import Algorithm, Sampling, ShearWarp
from vanilla_roll.rendering.composition import AccMax, AccMean, AccMin, AccVR, Composer
from vanilla_roll.rendering.mode import MIP, VR, Average, MinP, Mode
from vanilla_roll.rendering.orthogonal_sampling import (
    create_renderer as create_orthogonal_sampling,
)
from vanilla_roll.rendering.orthogonal_shear_warp import (
    create_renderer as create_orthogonal_shear_warp,
)
from vanilla_roll.rendering.projection import Orthogoal, Projection
from vanilla_roll.rendering.types import ColorImage, Image, MonoImage, Renderer
from vanilla_roll.volume import Volume


def _get_accumulator_constructor(
    rendering_mode: Mode,
    sampling_method: xpe.SamplingMethod = "linear",
) -> Callable[[tuple[int, int]], Composer]:
    match rendering_mode:
        case MinP():
            return lambda shape: AccMin(shape, sampling_method=sampling_method)
        case MIP():
            return lambda shape: AccMax(shape, sampling_method=sampling_method)
        case Average():
            return lambda shape: AccMean(shape, sampling_method=sampling_method)
        case VR(transfer_function):
            return lambda shape: AccVR(
                shape, transfer_function, sampling_method=sampling_method
            )
        case _:
            raise NotImplementedError(f"{rendering_mode}")


def create_renderer(
    volume: Volume,
    projection: Projection,
    rendering_method: Mode,
    sampling_method: xpe.SamplingMethod = "linear",
    algorithm: Algorithm = ShearWarp(),
) -> Renderer:
    accumulator_constructor = _get_accumulator_constructor(rendering_method)
    match (projection, algorithm):
        case (Orthogoal(), Sampling(step)):
            return create_orthogonal_sampling(
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


def convert_image_to_array(image: Image) -> xp.Array:
    match image:
        case MonoImage(l):
            return l
        case ColorImage(r, g, b):
            return xp.stack([r, g, b], axis=2)

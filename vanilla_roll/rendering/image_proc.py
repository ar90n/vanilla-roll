import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe
import vanilla_roll.array_api_image as xpi
from vanilla_roll.rendering.types import ColorImage, Image, MonoImage


def affine_image(
    image: Image,
    mat: xp.Array,
    output_shape: tuple[int, int],
    /,
    *,
    method: xpe.SamplingMethod = "linear",
) -> Image:
    match image:
        case MonoImage(l):
            return MonoImage(
                l=xpi.affine_transform(l, mat, output_shape, method=method)
            )
        case ColorImage(r, g, b):
            r = xpi.affine_transform(r, mat, output_shape, method=method)
            g = xpi.affine_transform(g, mat, output_shape, method=method)
            b = xpi.affine_transform(b, mat, output_shape, method=method)
            return ColorImage(r=r, g=g, b=b)


def resize_image(
    image: Image,
    output_shape: tuple[int, int],
    /,
    *,
    method: xpe.SamplingMethod = "linear",
) -> Image:
    match image:
        case MonoImage(l):
            return MonoImage(l=xpi.resize(l, output_shape, method=method))
        case ColorImage(r, g, b):
            r = xpi.resize(r, output_shape, method=method)
            g = xpi.resize(g, output_shape, method=method)
            b = xpi.resize(b, output_shape, method=method)
            return ColorImage(r=r, g=g, b=b)

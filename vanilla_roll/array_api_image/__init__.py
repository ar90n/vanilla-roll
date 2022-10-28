import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe


def affine_transform(
    image: xp.Array,
    mat: xp.Array,
    output_shape: tuple[int, int],
    /,
    *,
    method: xpe.SamplingMethod = "linear",
) -> xp.Array:
    iss, jss = xp.meshgrid(
        xp.arange(output_shape[1]), xp.arange(output_shape[0]), indexing="xy"
    )
    output_coords = xp.astype(
        xp.reshape(xp.stack([jss, iss, xp.ones_like(iss)], axis=2), (-1, 3)), xp.float32
    )
    input_coords = output_coords @ mat
    warped_pixels = xpe.sample(image, coordinates=input_coords[:, :2], method=method)
    return xp.reshape(warped_pixels, output_shape)


def resize(
    image: xp.Array,
    output_shape: tuple[int, int],
    /,
    *,
    method: xpe.SamplingMethod = "linear",
) -> xp.Array:
    mat = xp.asarray(
        [
            [image.shape[0] / output_shape[0], 0, 0],
            [0, image.shape[1] / output_shape[1], 0],
            [0, 0, 1],
        ]
    )
    return affine_transform(image, mat, output_shape, method=method)

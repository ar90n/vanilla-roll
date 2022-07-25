import vanilla_roll.array_api as xp
import vanilla_roll.array_api_extra as xpe


def affine_transform(
    image: xp.Array, mat: xp.Array, output_shape: tuple[int, int]
) -> xp.Array:
    iss, jss = xp.meshgrid(xp.arange(output_shape[1]), xp.arange(output_shape[0]))
    output_coords = xp.astype(
        xp.reshape(xp.stack([jss, iss, xp.ones_like(iss)], axis=2), (-1, 3)), xp.float64
    )
    input_coords = output_coords @ mat
    warped_pixels = xpe.sample(image, coordinates=input_coords[:, :2])
    return xp.reshape(warped_pixels, output_shape)

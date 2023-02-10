import numpy as np
import skimage.transform

import geodesic_shooting.core as core
from geodesic_shooting.utils import grid


def sample(f, coordinates, sampler_options={'order': 1, 'mode': 'edge'}):
    """Function to sample a given `ScalarFunction` or `VectorField` at given coordinates.

    Remark: The coordinates at which to sample have to be given as a `VectorField`.

    Parameters
    ----------
    f
        `ScalarFunction` or `VectorField` to transform.
    coordinates
        `VectorField` containing the coordinates to sample at.
    sampler_options
        Additional options passed to the `warp`-function,
        see https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp.

    Returns
    -------
    The sampled `ScalarFunction`, `VectorField` or `Diffeomorphism`.
    """
    assert (isinstance(f, (core.ScalarFunction, core.VectorField, core.Diffeomorphism))
            and isinstance(coordinates, core.Diffeomorphism))
    assert f.dim == coordinates.dim

    identity_grid = grid.coordinate_grid(f.spatial_shape).to_numpy()
    scaled_difference = np.einsum("...i,i->...i", coordinates.to_numpy() - identity_grid, f.spatial_shape)
    coordinates = np.einsum("...i->i...", identity_grid + scaled_difference)

    if isinstance(f, core.VectorField):
        samples_channels = []
        for i in range(f.dim):
            samples_channels.append(skimage.transform.warp(f[..., i], coordinates, **sampler_options))
        return core.VectorField(spatial_shape=f.spatial_shape, data=np.stack(samples_channels, axis=-1))
    elif isinstance(f, core.Diffeomorphism):
        samples_channels = []
        for i in range(f.dim):
            scaled_displacement_field = (f[..., i] - identity_grid[..., i]) / f.spatial_shape[i]
            samples_channels.append(skimage.transform.warp(scaled_displacement_field, coordinates, **sampler_options)
                                    + identity_grid[..., i])
        return core.Diffeomorphism(spatial_shape=f.spatial_shape, data=np.stack(samples_channels, axis=-1))

    transformed_function = skimage.transform.warp(f.to_numpy(), coordinates, **sampler_options)
    return core.ScalarFunction(spatial_shape=f.full_shape, data=transformed_function)

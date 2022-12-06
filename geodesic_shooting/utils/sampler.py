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
    The sampled `ScalarFunction` or `VectorField`.
    """
    assert isinstance(f, (core.ScalarFunction, core.VectorField)) and isinstance(coordinates, core.VectorField)
    assert f.dim == coordinates.dim

    coordinates = np.einsum("...i->i...", coordinates.to_numpy())

    if isinstance(f, core.VectorField):
        samples_channels = []
        for i in range(f.dim):
            samples_channels.append(skimage.transform.warp(f[..., i], coordinates, **sampler_options))
        return core.VectorField(spatial_shape=f.spatial_shape, data=np.stack(samples_channels, axis=-1))

    transformed_function = skimage.transform.warp(f.to_numpy(), coordinates, **sampler_options)
    return core.ScalarFunction(spatial_shape=f.full_shape, data=transformed_function)


def inverse_coordinates(coordinates, sampler_options={'order': 1, 'mode': 'edge'}):
    identity_grid = grid.coordinate_grid(coordinates.spatial_shape)
    return identity_grid + sample(identity_grid - coordinates, coordinates, sampler_options=sampler_options)


def sample_inverse(f, coordinates, sampler_options={'order': 1, 'mode': 'edge'}):
    inv_coordinates = inverse_coordinates(coordinates, sampler_options=sampler_options)
    return sample(f, inv_coordinates, sampler_options=sampler_options)

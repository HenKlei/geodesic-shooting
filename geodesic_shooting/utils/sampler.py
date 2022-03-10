import numpy as np
import skimage.transform

from geodesic_shooting.core import ScalarFunction, VectorField


def sample(f, coordinates, boundary_mode='edge'):
    """Function to sample a given input array at given coordinates.

    Parameters
    ----------
    array
        Input image.
    coordinates
        Array containing the coordinates to sample at.

    Returns
    -------
    The sampled array.
    """
    assert (isinstance(f, ScalarFunction) or isinstance(f, VectorField)) and isinstance(coordinates, VectorField)
    assert f.dim == coordinates.dim

    coordinates = np.einsum("...i->i...", coordinates.to_numpy())

    if isinstance(f, VectorField):
        samples_channels = []
        for i in range(f.dim):
            samples_channels.append(skimage.transform.warp(f[..., i], coordinates, mode=boundary_mode))
        return VectorField(f.spatial_shape, data=np.stack(samples_channels, axis=-1))

    return ScalarFunction(f.full_shape, skimage.transform.warp(f.to_numpy(), coordinates, mode=boundary_mode))

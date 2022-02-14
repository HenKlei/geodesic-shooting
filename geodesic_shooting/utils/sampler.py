import numpy as np
import skimage.transform


def sample(array, coordinates):
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
    assert array.ndim in [coordinates.ndim, coordinates.ndim - 1]

    if coordinates.ndim == 1:
        return skimage.transform.warp(array, coordinates[np.newaxis, ...], mode='edge')

    if array.ndim == coordinates.ndim:
        samples_channels = []
        for i in range(array.shape[0]):
            samples_channels.append(skimage.transform.warp(array[i], coordinates, mode='edge'))
        return np.stack(samples_channels, axis=0)

    return skimage.transform.warp(array, coordinates, mode='edge')

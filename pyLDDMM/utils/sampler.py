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
    assert array.ndim in [1, 2, 3]
    assert coordinates.ndim in [1, 2, 3]

    if coordinates.ndim == 2:
        if array.ndim == 1:
            return skimage.transform.warp(array, coordinates, mode='edge')
        return skimage.transform.warp(array[0, :], coordinates, mode='edge')

    # add newaxis to coordinates if coordinates has only a single dimension
    if coordinates.ndim == 1:
        return skimage.transform.warp(array, coordinates[np.newaxis, ...], mode='edge')

    # only a single color channel
    if array.ndim == 2:
        return skimage.transform.warp(array, coordinates, mode='edge')

    if array.ndim == 3:
        # the first dimension is the channel dimension,
        # we need iterate over each channel independently
        samples_channels = []
        for i in range(array.shape[0]):
            samples_channels.append(skimage.transform.warp(array[i, :, :], coordinates, mode='edge'))
        return np.stack(samples_channels, axis=0)

    raise NotImplementedError


if __name__ == "__main__":
    from pyLDDMM.utils import grid

    shape = (5, 10)
    input_coordinates = grid.coordinate_grid(shape)
    array1 = np.random.rand(*shape)

    # single color channel...
    result1 = sample(array1, input_coordinates)
    assert (array1 == result1).all()

    array2 = np.random.rand(*shape)
    input_array = np.stack([array1, array2], axis=0)

    # two color channels...
    result = sample(input_array, input_coordinates)
    assert (input_array == result).all()

import numpy as np
import skimage.transform


def sample(array, coordinates):
    """
    samples the array at given coordinates
    @param array: image array of shape n x H x W or H x W
    @param coordinates: array of shape 2 x H x W
    @return:
    """
    assert array.ndim in [1, 2, 3]
    assert coordinates.ndim in [1, 2, 3]

    # Reshape coordinate for skimage.
    if coordinates.ndim == 2:
        if array.ndim == 1:
            return skimage.transform.warp(array, coordinates, mode='edge')
        return skimage.transform.warp(array[0, :], coordinates, mode='edge')

    if coordinates.ndim == 1:
        return skimage.transform.warp(array, coordinates[np.newaxis, ...], mode='edge')

    if array.ndim == 2:
        # Only a single color channel. Go ahead...
        return skimage.transform.warp(array, coordinates, mode='edge')

    if array.ndim == 3:
        # The first dimension is the channel dimension.
        # We need to sample each channel independently.
        C = array.shape[0]
        samples_channels = []
        for c in range(C):
            samples_channels.append(skimage.transform.warp(array[c, :, :], coordinates, mode='edge'))
        return np.stack(samples_channels, axis=0)

    raise NotImplementedError


if __name__ == "__main__":
    from pyLDDMM.utils import grid

    shape = (5, 10)
    coordinates = grid.coordinate_grid(shape)
    array1 = np.random.rand(*shape)

    # Single color channel...
    result1 = sample(array1, coordinates)
    assert (array1 == result1).all()

    array2 = np.random.rand(*shape)
    array = np.stack([array1, array2], axis=0)

    # Two color channels...
    result = sample(array, coordinates)
    assert (array == result).all()

import skimage.transform
import numpy as np

def sample(array, coordinates):
    """
    samples the array at given coordinates
    @param array: image array of shape H x W x n or H x W
    @param coordinates: array of shape H x W x 2
    @return:
    """

    assert array.ndim in [1, 2, 3]

    # reshape coordinate for skimage
    if coordinates.ndim == 2:
        coordinates = np.transpose(coordinates)
        if array.ndim == 1:
            return skimage.transform.warp(array, coordinates, mode='edge')
        return np.transpose([skimage.transform.warp(array[:, 0], coordinates, mode='edge'),])

    if coordinates.ndim == 3:
        coordinates = np.transpose(coordinates, axes=[2, 1, 0])

    if array.ndim == 2:
        # only a single color channel. go ahead
        return skimage.transform.warp(array, coordinates, mode='edge')
    elif array.ndim == 3:
        # the last dimension is the channel dimension. We need to sample each channel independently.
        C = array.shape[-1]
        samples_channels = []
        for c in range(C):
            samples_channels.append(skimage.transform.warp(array[:, :, c], coordinates, mode='edge'))
        return np.stack(samples_channels, axis=-1)

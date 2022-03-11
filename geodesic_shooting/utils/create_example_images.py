import numpy as np

from geodesic_shooting.core import ScalarFunction


def make_circle(shape, center, radius):
    """Creates a square image with a circle in it.

    Parameters
    ----------
    shape
        Shape of the resulting image (number of pixels in the two directions of the image).
    center
        Center of the circle (in pixels).
    radius
        Radius of the circle (in pixels).

    Returns
    -------
    The image as numpy array.
    """
    l = [np.arange(s) for s in shape]
    XX, YY = np.meshgrid(*l, indexing='ij')
    XY = np.stack([XX, YY], axis=-1)
    val = np.linalg.norm(XY - center[np.newaxis, np.newaxis, :], axis=-1)
    result = np.zeros(shape)
    result += 1. * (val < radius)
    return ScalarFunction(spatial_shape=shape, data=result)


def make_square(shape, center, length):
    """Creates a square image with a square in it.

    Parameters
    ----------
    shape
        Shape of the resulting image (number of pixels in the two directions of the image).
    center
        Center of the square (in pixels).
    length
        Length of each side of the square.

    Returns
    -------
    The image as numpy array.
    """
    l = [np.arange(s) for s in shape]
    XX, YY = np.meshgrid(*l, indexing='ij')
    XY = np.stack([XX, YY], axis=-1)
    val = np.linalg.norm(XY - center[np.newaxis, np.newaxis, :], axis=-1, ord=np.inf)
    result = np.zeros(shape)
    result += 1. * (val < length / 2.)
    return ScalarFunction(spatial_shape=shape, data=result)

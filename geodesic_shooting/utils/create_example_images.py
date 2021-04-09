import numpy as np


def make_circle(N, center, radius):
    """Creates a square image with a circle in it.

    Parameters
    ----------
    N
        Number of pixels in both directions of the image.
    center
        Center of the circle (in pixels).
    radius
        Radius of the circle (in pixels).

    Returns
    -------
    The image as numpy array.
    """
    XX, YY = np.meshgrid(np.arange(N), np.arange(N))
    XY = np.stack([XX, YY], axis=-1)
    val = np.linalg.norm(XY - center[np.newaxis, np.newaxis, :], axis=2)
    result = np.zeros((N, N))
    result += 1. * (val < radius)
    return result


def make_square(N, center, length):
    """Creates a square image with a square in it.

    Parameters
    ----------
    N
        Number of pixels in both directions of the image.
    center
        Center of the square (in pixels).
    length
        Length of each side of the square.

    Returns
    -------
    The image as numpy array.
    """
    XX, YY = np.meshgrid(np.arange(N), np.arange(N))
    XY = np.stack([XX, YY], axis=-1)
    val = np.linalg.norm(XY - center[np.newaxis, np.newaxis, :], axis=2, ord=np.inf)
    result = np.zeros((N, N))
    result += 1. * (val < length / 2.)
    return result

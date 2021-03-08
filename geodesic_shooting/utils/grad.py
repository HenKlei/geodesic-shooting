import numpy as np
from scipy.ndimage import convolve


def finite_difference(array):
    """Finite difference scheme for approximating the derivative of the input array.

    This function uses central differences to compute the (discrete) derivative
    of the input array in the different dimensions.

    Parameters
    ----------
    array
        Input array to compute the derivative of.

    Returns
    -------
    Array containing the derivatives in the different dimensions.
    """
    # mind the unusual ordering due to the usage of convolutions instead of correlations
    window = np.array([1., 0., -1.])
    dim = array.ndim
    window = window.reshape(list(window.shape)+[1,]*(dim-1)).T

    derivatives = []
    for d in range(dim):
        indices = list(range(dim))
        indices[0] = d
        indices[d] = 0
        window_d = np.transpose(window, axes=indices)
        derivative_d = convolve(array, window_d)
        derivatives.append(derivative_d)

    return np.flip(np.stack(derivatives, axis=0), axis=0)


if __name__ == "__main__":
    img = np.zeros((5, 10))
    img[..., 2] = 1
    derivative = np.zeros((2, 5, 10))
    derivative[1, :, 1] = 1
    derivative[1, :, 3] = -1
    assert (finite_difference(img) == derivative).all()

    img = np.zeros((5, 10))
    img[2, ...] = 1
    derivative = np.zeros((2, 5, 10))
    derivative[0, 1, :] = 1
    derivative[0, 3, :] = -1
    assert (finite_difference(img) == derivative).all()

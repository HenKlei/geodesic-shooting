import numpy as np
from scipy.ndimage import correlate
from scipy.linalg import toeplitz


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
    window = np.array([-1., 0., 1.])
    dim = array.ndim
    window = window.reshape(list(window.shape) + [1, ]*(dim-1)).T

    derivatives = []
    for d in range(dim):
        indices = list(range(dim))
        indices[0] = d
        indices[d] = 0
        window_d = np.transpose(window, axes=indices)
        derivative_d = correlate(array, window_d)
        derivatives.append(derivative_d)

    return np.flip(np.stack(derivatives, axis=0), axis=0)[0:array.shape[0], ...]


def finite_difference_matrix(dim, size):
    assert dim in [1, ]

    if dim == 1:
        column = np.zeros(size)
        column[1] = -1
        row = np.zeros(size)
        row[1] = 1
        mat = toeplitz(column, row)

    assert mat.shape == (size, size)
    return mat


if __name__ == "__main__":
    img = np.zeros(10)
    img[2] = 1
    derivative = np.zeros(10)
    derivative[1] = 1
    derivative[3] = -1
    assert (finite_difference(img) == derivative).all()
    assert (finite_difference_matrix(1, 10).dot(img) == derivative).all()

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

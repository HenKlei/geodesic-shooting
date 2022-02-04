import numpy as np
from scipy.ndimage import correlate

from geodesic_shooting.utils.helper_functions import tuple_product


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
    window = np.array([-1., 0., 1.]) / 2.
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


def finite_difference_matrix(shape):
    dim = len(shape)
    assert dim in [1, 2, ]
    size = tuple_product(shape)

    if dim == 1:
        main_diagonal = np.zeros(size)
        main_diagonal[0] = -0.5
        main_diagonal[-1] = 0.5
        first_diagonal = np.ones(size-1) * 0.5
        mat = np.diag(main_diagonal, 0) + np.diag(first_diagonal, 1) + np.diag(-first_diagonal, -1)

    if dim == 2:
        main_diagonal = np.zeros(size)
        main_diagonal[0:shape[1]] = -0.5
        main_diagonal[-shape[1]:] = 0.5
        first_diagonal = np.ones(size-shape[1]) * 0.5
        mat1 = (np.diag(main_diagonal, 0) + np.diag(first_diagonal, shape[1])
                + np.diag(-first_diagonal, -shape[1]))

        main_diagonal = np.zeros(size)
        main_diagonal[::shape[1]] = -0.5
        main_diagonal[shape[1]-1::shape[1]] = 0.5
        first_diagonal = np.ones(size-1) * 0.5
        first_diagonal[shape[1]-1::shape[1]] = 0
        mat2 = (np.diag(main_diagonal, 0) + np.diag(first_diagonal, 1)
                + np.diag(-first_diagonal, -1))

        mat = np.vstack([np.hstack([mat1, mat2]), np.hstack([mat1, mat2])])

    assert mat.shape == (dim * size, dim * size)
    return mat


def gradient_matrix(shape):
    dim = len(shape)
    assert dim in [1, 2, ]
    size = tuple_product(shape)

    if dim == 1:
        main_diagonal = np.zeros(size)
        main_diagonal[0] = -0.5
        main_diagonal[-1] = 0.5
        first_diagonal = np.ones(size-1) * 0.5
        mat = np.diag(main_diagonal, 0) + np.diag(first_diagonal, 1) + np.diag(-first_diagonal, -1)

    if dim == 2:
        main_diagonal = np.zeros(size)
        main_diagonal[0:shape[1]] = -0.5
        main_diagonal[-shape[1]:] = 0.5
        first_diagonal = np.ones(size-shape[1]) * 0.5
        mat1 = (np.diag(main_diagonal, 0) + np.diag(first_diagonal, shape[1])
                + np.diag(-first_diagonal, -shape[1]))

        main_diagonal = np.zeros(size)
        main_diagonal[::shape[1]] = -0.5
        main_diagonal[shape[1]-1::shape[1]] = 0.5
        first_diagonal = np.ones(size-1) * 0.5
        first_diagonal[shape[1]-1::shape[1]] = 0
        mat2 = (np.diag(main_diagonal, 0) + np.diag(first_diagonal, 1)
                + np.diag(-first_diagonal, -1))

        mat = np.vstack([mat1, mat2])

    assert mat.shape == (dim * size, size)
    return mat


def divergence_matrix(shape):
    dim = len(shape)
    size = tuple_product(shape)

    div_tilde = np.zeros((size, dim * size))
    assert div_tilde.shape == (size, dim * size)
    div = np.vstack([div_tilde, ] * dim)
    assert div.shape == (dim * size, dim * size)
    return div

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
        mat1 = np.diag(main_diagonal, 0) + np.diag(first_diagonal, 1) + np.diag(-first_diagonal, -1)
        mat = np.array([mat1, ])

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

        mat = np.array([mat1, mat2])

    assert mat.shape == (dim, size, size)
    return mat


def divergence_matrix(shape):
    return np.sum(finite_difference_matrix(shape), axis=0)


if __name__ == "__main__":
    img = np.zeros(10)
    img[2] = 1
    derivative = np.zeros(10)
    derivative[1] = 0.5
    derivative[3] = -0.5
    assert (finite_difference(img) == derivative).all()
    assert (finite_difference_matrix(img.shape).dot(img) == derivative).all()

    img = np.arange(10)
    derivative_fd = np.ones(10)
    derivative_fd_mat = np.ones(10)
    derivative_fd[0] = 0
    derivative_fd[-1] = 0
    derivative_fd_mat[0] = 0.5
    derivative_fd_mat[-1] = 0.5
    assert (finite_difference(img) == derivative_fd).all()
    assert (finite_difference_matrix(img.shape).dot(img) == derivative_fd_mat).all()
    divergence_mat = derivative_fd_mat
    assert (divergence_matrix(img.shape).dot(img) == divergence_mat).all()

    img = np.zeros((5, 10))
    img = np.zeros((3, 5))
    img[..., 2] = 1
    derivative = np.zeros((2, *img.shape))
    derivative[1, :, 1] = 0.5
    derivative[1, :, 3] = -0.5
    assert (finite_difference(img) == derivative).all()
    assert (finite_difference_matrix(img.shape).dot(img.flatten()).reshape((img.ndim, *img.shape))
            == derivative).all()
    divergence_mat = derivative[1]
    assert (divergence_matrix(img.shape).dot(img.flatten()).reshape(img.shape)
            == divergence_mat).all()

    img = np.zeros((5, 10))
    img[2, ...] = 1
    derivative = np.zeros((2, 5, 10))
    derivative[0, 1, :] = 0.5
    derivative[0, 3, :] = -0.5
    assert (finite_difference(img) == derivative).all()
    assert (finite_difference_matrix(img.shape).dot(img.flatten()).reshape((img.ndim, *img.shape))
            == derivative).all()
    divergence_mat = derivative[0]
    assert (divergence_matrix(img.shape).dot(img.flatten()).reshape(img.shape)
            == divergence_mat).all()

    img = np.zeros((5, 10))
    img[2, ...] = 1
    img[..., 2] = 1
    divergence_mat = np.zeros((5, 10))
    divergence_mat[:, 1] = 0.5
    divergence_mat[:, 3] = -0.5
    divergence_mat[1, :] += 0.5
    divergence_mat[3, :] += -0.5
    divergence_mat[1, 2] = 0.
    divergence_mat[2, 1] = 0.
    divergence_mat[3, 2] = 0.
    divergence_mat[2, 3] = 0.
    assert (np.sum(finite_difference_matrix(img.shape).dot(img.flatten())
                   .reshape((img.ndim, *img.shape)), axis=0) == divergence_mat).all()
    assert (divergence_matrix(img.shape).dot(img.flatten()).reshape(img.shape)
            == divergence_mat).all()

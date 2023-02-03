import numpy as np
from scipy.ndimage import correlate

import geodesic_shooting.core as core
from geodesic_shooting.utils.helper_functions import tuple_product


def finite_difference(f):
    """Finite difference scheme for approximating the derivative of the input.

    This function uses central differences to compute the (discrete) derivative
    of the input in the different dimensions.

    Parameters
    ----------
    f
        `ScalarFunction` or `VectorField` to compute the derivative of.

    Returns
    -------
    In the case of a `ScalarFunction` `f`, the result is a gradient `VectorField`,
    in the case of a `VectorField` `f`, the result is a numpy-array containing the
    gradient/Jacobian of the `VectorField` at the spatial points.
    """
    assert isinstance(f, (core.ScalarFunction, core.VectorField))

    window = np.array([-1., 0., 1.]) * 0.5
    dim = f.dim
    window = window.reshape(list(window.shape) + [1, ]*(dim-1))
    shape = f.full_shape

    def _fd_single_dim(u, d):
        indices = list(range(dim))
        indices[0] = d
        indices[d] = 0
        window_d = np.transpose(window / shape[d], axes=indices)
        return correlate(u, window_d)

    derivatives = []

    if isinstance(f, core.VectorField):
        for i in range(f.dim):
            derivatives_d = []
            for j in range(f.dim):
                derivatives_d.append(_fd_single_dim(f[..., i], j))
            derivatives.append(np.stack(derivatives_d, axis=-1))
        return np.stack(derivatives, axis=-1)

    if isinstance(f, core.ScalarFunction):
        for d in range(dim):
            derivatives.append(_fd_single_dim(f.to_numpy(), d))
        return core.VectorField(spatial_shape=f.spatial_shape, data=np.stack(derivatives, axis=-1))

    raise NotImplementedError


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

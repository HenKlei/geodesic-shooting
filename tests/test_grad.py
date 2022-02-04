import numpy as np

from geodesic_shooting.utils.grad import finite_difference, finite_difference_matrix, gradient_matrix


def test_grad():
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

    img = np.zeros((3, 5))
    img[..., 2] = 1
    derivative = np.zeros((2, *img.shape))
    derivative[1, :, 1] = 0.5
    derivative[1, :, 3] = -0.5
    assert (finite_difference(img) == derivative).all()
    assert (gradient_matrix(img.shape).dot(img.flatten()).reshape((img.ndim, *img.shape)) == derivative).all()

    img = np.zeros((5, 10))
    img[2, ...] = 1
    derivative = np.zeros((2, 5, 10))
    derivative[0, 1, :] = 0.5
    derivative[0, 3, :] = -0.5
    assert (finite_difference(img) == derivative).all()
    assert (gradient_matrix(img.shape).dot(img.flatten()).reshape((img.ndim, *img.shape))
            == derivative).all()

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
    assert (np.sum(gradient_matrix(img.shape).dot(img.flatten())
                   .reshape((img.ndim, *img.shape)), axis=0) == divergence_mat).all()

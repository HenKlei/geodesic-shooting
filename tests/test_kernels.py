import numpy as np
import pytest

import geodesic_shooting
from geodesic_shooting.utils.kernels import GaussianKernel, RationalQuadraticKernel


@pytest.mark.parametrize("sigma", [2. * (1 - x) for x in np.random.rand(10)])
def test_gaussian_kernel(sigma):
    k = GaussianKernel(sigma=sigma)
    assert np.allclose(k(np.array([5., 3.]), np.array([4., 2.])),
                       np.exp(-sigma * 2.) * np.eye(2))
    assert np.allclose(k(np.array([5., 3.]), np.array([4., 2.])),
                       k(np.array([4., 2.]), np.array([5., 3.])))


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("num_positions", [1, 2, 5, 10, 20])
@pytest.mark.parametrize("sigma", [2. * (1 - x) for x in np.random.rand(10)])
def test_gaussian_kernel_vectorized_application(dim, num_positions, sigma):
    k = GaussianKernel(sigma=sigma)
    positions = np.random.rand(dim * num_positions)
    positions2 = np.random.rand(dim * num_positions)
    full_kernel_matrix = k.apply_vectorized(positions, positions2, dim)
    assert all([np.allclose(full_kernel_matrix[i*dim:(i+1)*dim, j*dim:(j+1)*dim],
                            k(positions[i*dim:(i+1)*dim], positions2[j*dim:(j+1)*dim]))
                for i in range(num_positions) for j in range(num_positions)])

    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': sigma},
                                            dim=dim, num_landmarks=num_positions)
    assert all([np.allclose(k(positions[i*dim:(i+1)*dim], positions[j*dim:(j+1)*dim]),
                            gs.K(positions)[i*dim:(i+1)*dim, j*dim:(j+1)*dim])
                for i in range(num_positions) for j in range(num_positions)])


@pytest.mark.parametrize("sigma", [2. * (1 - x) for x in np.random.rand(10)])
def test_gaussian_kernel_derivatives(sigma):
    k = GaussianKernel(sigma=sigma)
    assert np.allclose(k.derivative_1(np.array([5., 3.]), np.array([3., 2.]), 0),
                       -2. * 2. * sigma * np.exp(-5. * sigma) * np.eye(2))
    assert np.allclose(k.derivative_1(np.array([5., 3.]), np.array([3., 2.]), 1),
                       -2. * sigma * np.exp(-5. * sigma) * np.eye(2))

    assert np.allclose(k.derivative_2(np.array([5., 3.]), np.array([3., 2.]), 0),
                       2. * 2. * sigma * np.exp(-5. * sigma) * np.eye(2))
    assert np.allclose(k.derivative_2(np.array([5., 3.]), np.array([3., 2.]), 1),
                       2. * sigma * np.exp(-5. * sigma) * np.eye(2))

    h = 1e-8
    tol = 1e-7
    assert np.linalg.norm((k(np.array([5. + h, 3.]), np.array([3., 2.]))[0][0]
                           - k(np.array([5., 3.]), np.array([3., 2.]))[0][0]) / h
                          - k.derivative_1(np.array([5., 3.]), np.array([3., 2.]), 0)[0, 0]) < tol

    assert np.allclose(k.full_derivative_1(np.array([5., 3.]), np.array([3., 2.])),
                       np.array([[[-2. * 2. * sigma * np.exp(-5. * sigma), 0], [0, -2. * 2. * sigma * np.exp(-5. * sigma)]],
                                 [[-2. * sigma * np.exp(-5. * sigma), 0], [0, -2. * sigma * np.exp(-5. * sigma)]]]))


def test_rational_quadratic_kernel():
    k = RationalQuadraticKernel(sigma=1.)
    assert np.isclose(np.linalg.norm(k(np.array([5., 3.]), np.array([4., 2.]))
                                     - np.eye(2) / 3.), 0.0)
    assert np.isclose(np.linalg.norm(k(np.array([5., 3.]), np.array([4., 2.]))
                                     - k(np.array([4., 2.]), np.array([5., 3.]))), 0.0)

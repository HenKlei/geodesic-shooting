import numpy as np
import pytest

from geodesic_shooting.utils.kernels import GaussianKernel, RationalQuadraticKernel


@pytest.mark.parametrize("sigma", [2. * (1 - x) for x in np.random.rand(10)])
def test_gaussian_kernel(sigma):
    k = GaussianKernel(sigma=sigma)
    assert np.allclose(k(np.array([5., 3.]), np.array([4., 2.])),
                       np.exp(-sigma * 2.) * np.eye(2))
    assert np.allclose(k(np.array([5., 3.]), np.array([4., 2.])),
                       k(np.array([4., 2.]), np.array([5., 3.])))


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

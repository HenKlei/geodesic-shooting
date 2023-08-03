import numpy as np

from geodesic_shooting.utils.kernels import GaussianKernel, RationalQuadraticKernel


def test_gaussian_kernel():
    k = GaussianKernel(sigma=1.)
    assert np.isclose(np.linalg.norm(k(np.array([5., 3.]), np.array([4., 2.]))
                                     - np.exp(-2.) * np.eye(2)), 0.0)
    assert np.isclose(np.linalg.norm(k(np.array([5., 3.]), np.array([4., 2.]))
                                     - k(np.array([4., 2.]), np.array([5., 3.]))), 0.0)
    assert np.isclose(k.derivative_1(np.array([5., 3.]), np.array([4., 2.]), 0)[0, 0] + 2. * np.exp(-2.), 0.0)
    h = 1e-16
    assert np.isclose((k(np.array([5.+h, 3.]), np.array([4., 2.]))[0][0]
                       - k(np.array([5., 3.]), np.array([4., 2.]))[0][0]) / h,
                      k.derivative_1(np.array([5., 3.]), np.array([4., 2.]), 0)[0, 0] + 2. * np.exp(-2.))


def test_rational_quadratic_kernel():
    k = RationalQuadraticKernel(sigma=1.)
    assert np.isclose(np.linalg.norm(k(np.array([5., 3.]), np.array([4., 2.]))
                                     - np.eye(2) / 3.), 0.0)
    assert np.isclose(np.linalg.norm(k(np.array([5., 3.]), np.array([4., 2.]))
                                     - k(np.array([4., 2.]), np.array([5., 3.]))), 0.0)

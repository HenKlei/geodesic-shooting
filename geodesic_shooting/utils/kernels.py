import numpy as np


class Kernel:
    """Base class for kernels."""
    def __call__(self, x, y):
        raise NotImplementedError


class GaussianKernel(Kernel):
    """Class that implements a Gaussian (matrix-valued, diagonal) kernel."""
    def __init__(self, sigma=1./np.sqrt(2.)):
        """Constructor.

        Parameters
        ----------
        sigma
            Scaling parameter for the Gaussian bell curve.
        """
        super().__init__()
        assert sigma > 0
        self.sigma = sigma

    def __call__(self, x, y):
        assert x.ndim == 1
        assert x.shape == y.shape
        return np.exp(-np.linalg.norm(x-y)**2 / (2*self.sigma**2)) * np.eye(x.shape[0])

    def derivative_1(self, x, y, i):
        """Derivative of kernel with respect to i-th component of x."""
        assert x.ndim == 1
        assert x.shape == y.shape
        assert 0 <= i < x.shape[0]
        return (y[i] - x[i]) / self.sigma**2 * self(x, y)[0][0]

    def derivative_2(self, x, y, i):
        """Derivative of kernel with respect to i-th component of y."""
        assert x.ndim == 1
        assert x.shape == y.shape
        assert 0 <= i < x.shape[0]
        return (x[i] - y[i]) / self.sigma**2 * self(x, y)[0][0]


k = GaussianKernel(sigma=1.)
assert np.isclose(np.linalg.norm(k(np.array([5., 3.]), np.array([4., 2.]))
                                 - np.exp(-1.) * np.eye(2)), 0.0)
assert np.isclose(k.derivative_1(np.array([5., 3.]), np.array([4., 2.]), 0) + np.exp(-1.), 0.0)
h = 1e-16
assert np.isclose((k(np.array([5.+h, 3.]), np.array([4., 2.]))[0][0]
                   - k(np.array([5., 3.]), np.array([4., 2.]))[0][0]) / h,
                  k.derivative_1(np.array([5., 3.]), np.array([4., 2.]), 0) + np.exp(-1.))

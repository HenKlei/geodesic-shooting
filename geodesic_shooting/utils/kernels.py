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
        return np.exp(-np.linalg.norm(x-y)**2 / (self.sigma**2)) * np.eye(x.shape[0]) / (self.sigma**2)

    def derivative_1(self, x, y, i):
        """Derivative of kernel with respect to i-th component of x."""
        assert x.ndim == 1
        assert x.shape == y.shape
        assert 0 <= i < x.shape[0]
        return (y[i] - x[i]) / (self.sigma**4) * self(x, y)[0][0]

    def derivative_2(self, x, y, i):
        """Derivative of kernel with respect to i-th component of y."""
        assert x.ndim == 1
        assert x.shape == y.shape
        assert 0 <= i < x.shape[0]
        return (x[i] - y[i]) / (self.sigma**4) * self(x, y)[0][0]

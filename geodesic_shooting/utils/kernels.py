import numpy as np


class Kernel:
    """Base class for kernels."""
    def __call__(self, x, y):
        raise NotImplementedError


class GaussianKernel(Kernel):
    """Class that implements a Gaussian (matrix-valued, diagonal) kernel."""
    def __init__(self, scalar=False, sigma=1./np.sqrt(2.)):
        """Constructor.

        Parameters
        ----------
        scalar
            If `True`, a scalar-valued kernel is returned, otherwise a matrix-valued,
            diagonal kernel.
        sigma
            Scaling parameter for the Gaussian bell curve.
        """
        super().__init__()
        assert sigma > 0
        self.scalar = scalar
        self.sigma = sigma

    def __call__(self, x, y):
        assert x.ndim == 1
        assert x.shape == y.shape
        res = np.exp(-np.linalg.norm(x-y)**2 / (2. * self.sigma**2))
        if self.scalar:
            return res
        else:
            return res * np.eye(x.shape[0])

    def derivative_1(self, x, y, i):
        """Derivative of kernel with respect to i-th component of x."""
        assert x.ndim == 1
        assert x.shape == y.shape
        assert 0 <= i < x.shape[0]
        res = (y[i] - x[i]) / self.sigma**2
        if self.scalar:
            return res * self(x, y)
        else:
            return res * self(x, y)[0][0]

    def derivative_2(self, x, y, i):
        """Derivative of kernel with respect to i-th component of y."""
        assert x.ndim == 1
        assert x.shape == y.shape
        assert 0 <= i < x.shape[0]
        return -self.derivative_1(x, y, i)


class RationalQuadraticKernel(Kernel):
    """Class that implements a rational quadratic (matrix-valued, diagonal) kernel."""
    def __init__(self, scalar=False, sigma=1./np.sqrt(2.), alpha=1):
        """Constructor.

        Parameters
        ----------
        scalar
            If `True`, a scalar-valued kernel is returned, otherwise a matrix-valued,
            diagonal kernel.
        sigma
            Scaling parameter for the squared norm.
        alpha
            Exponent of the denominator.
        """
        super().__init__()
        assert sigma > 0
        assert alpha > 0
        self.scalar = scalar
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, x, y):
        assert x.ndim == 1
        assert x.shape == y.shape
        res = 1. / ((1. + np.linalg.norm(x-y)**2 / (self.sigma**2))**self.alpha)
        if self.scalar:
            return res
        else:
            return res * np.eye(x.shape[0])

    def derivative_1(self, x, y, i):
        """Derivative of kernel with respect to i-th component of x."""
        assert x.ndim == 1
        assert x.shape == y.shape
        assert 0 <= i < x.shape[0]
        return 2. * self.alpha * (y[i] - x[i]) / (self.sigma**2
                                                  * ((1. + np.linalg.norm(x-y)**2 / (self.sigma**2))**(self.alpha + 1)))

    def derivative_2(self, x, y, i):
        """Derivative of kernel with respect to i-th component of y."""
        assert x.ndim == 1
        assert x.shape == y.shape
        assert 0 <= i < x.shape[0]
        return -self.derivative_1(x, y, i)

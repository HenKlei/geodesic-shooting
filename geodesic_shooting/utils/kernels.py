import numpy as np
from scipy.spatial import distance_matrix


class Kernel:
    """Base class for kernels."""
    def __str__(self):
        return f"{self.__class__.__name__}"

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

    def __str__(self):
        return f"{self.__class__.__name__}: sigma={self.sigma}"

    def __call__(self, x, y):
        if x.ndim == 1:
            x = x.reshape((1, -1))
        if y.ndim == 1:
            y = y.reshape((1, -1))
        assert x.ndim == 2
        assert y.ndim == 2
        res = np.exp(-(self.sigma * distance_matrix(np.atleast_2d(x), np.atleast_2d(y))) ** 2)
        if self.scalar:
            return res
        else:
            return np.kron(res, np.eye(x.shape[1]))

    def derivative_1(self, x, y):
        """Derivative of kernel with respect to i-th component of x."""
        assert x.ndim == 1
        assert x.shape == y.shape
        res = (y - x) / self.sigma**2
        res_self = self(x, y)
        return np.kron(res, res_self).T.reshape((res.shape[0], *res_self.shape)).swapaxes(0, 1)

    def derivative_2(self, x, y):
        """Derivative of kernel with respect to i-th component of y."""
        assert x.ndim == 1
        assert x.shape == y.shape
        return -self.derivative_1(x, y)


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

    def __str__(self):
        return f"{self.__class__.__name__}: sigma={self.sigma}, alpha={self.alpha}"

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

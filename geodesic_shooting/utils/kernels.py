import numpy as np

from scipy.spatial import distance_matrix


class RBFKernel:
    """Base class for kernels."""
    def __init__(self, scalar):
        self.scalar = scalar

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __call__(self, x, y):
        assert x.ndim == 1
        assert x.shape == y.shape
        res = self._apply_rbf(np.linalg.norm(x - y))
        if self.scalar:
            return res
        else:
            return res * np.eye(x.shape[0])

    def apply_vectorized(self, x, y, dim):
        if x.ndim == 1:
            x = x.reshape((-1, dim))
        if y.ndim == 1:
            y = y.reshape((-1, dim))
        num_elements_x = x.shape[ 0]
        num_elements_y = y.shape[0]

        dist_mat = distance_matrix(x, y)
        res = self._apply_rbf(dist_mat)
        assert res.shape == (num_elements_x, num_elements_y)
        if self.scalar:
            return res
        else:
            res = np.einsum('ij,kl->ijkl', res, np.eye(dim))
            assert res.shape == (num_elements_x, num_elements_y, dim, dim)
            res = res.swapaxes(1, 2).reshape((num_elements_x * dim,num_elements_y * dim))
            return res

    def _apply_rbf(self, d):
        raise NotImplementedError


class GaussianKernel(RBFKernel):
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
        super().__init__(scalar)

        assert sigma > 0
        self.sigma = sigma

    def __str__(self):
        return f"{self.__class__.__name__}: sigma={self.sigma}"

    def _apply_rbf(self, d):
        return np.exp(-d ** 2 / (2. * self.sigma ** 2))

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


class RationalQuadraticKernel(RBFKernel):
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
        super().__init__(scalar)

        assert sigma > 0
        assert alpha > 0
        self.sigma = sigma
        self.alpha = alpha

    def __str__(self):
        return f"{self.__class__.__name__}: sigma={self.sigma}, alpha={self.alpha}"

    def _apply_rbf(self, d):
        return 1. / ((1. + d**2 / (self.sigma**2))**self.alpha)

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

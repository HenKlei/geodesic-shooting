import numpy as np
from scipy.ndimage import convolve
import scipy.sparse as sp
from scipy.sparse.linalg import inv as spinv

from geodesic_shooting.utils.helper_functions import tuple_product, fftn, ifftn


class BiharmonicRegularizer:
    """Biharmonic regularizer implementing smoothing functions.

    This class implements a regularizer for vector fields to make them smooth,
    such that the corresponding flows define diffeomorphisms.
    """
    def __init__(self, alpha=1, exponent=1):
        """Constructor.

        Parameters
        ----------
        alpha
            Smoothness parameter that determines how strong the smoothing effect should be.
        exponent
            Raises the smoothing operator to a certain power.
        """
        assert alpha > 0
        assert isinstance(exponent, int) and exponent > 0

        self.alpha = alpha
        self.exponent = exponent

        self.helper_operator = None

        self.cauchy_navier_matrix = None
        self.cauchy_navier_inverse_matrix = None
        self.cauchy_navier_squared_inverse_matrix = None

    def init_matrices(self, shape):
        """Initializes the Cauchy-Navier operator matrix and inverse matrices.

        It is very time-consuming to compute the inverse, but solving many linear
        systems of equations in each iteration is costly as well.

        Parameters
        ----------
        shape
            Shape of the input images.
        """
        self.cauchy_navier_matrix = self._cauchy_navier_matrix(shape)
        self.cauchy_navier_inverse_matrix = spinv(self.cauchy_navier_matrix)
        self.cauchy_navier_squared_inverse_matrix = self.cauchy_navier_inverse_matrix.dot(
            self.cauchy_navier_inverse_matrix)

    def cauchy_navier(self, function):
        """Application of the Cauchy-Navier type operator (-alpha * Δ + I) to a function.

        Parameters
        ----------
        function
            Array that holds the function values.

        Returns
        -------
        Array of the same shape as the input.
        """
        assert function.ndim in [2, 3]

        # window depends on the dimension and represents a simple approximation
        # of the Laplace operator
        if function.ndim == 2:
            window = np.array([1., -2., 1.])
        elif function.ndim == 3:
            window = np.array([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]])

        dff = np.stack([convolve(function[d, ...], window)
                        for d in range(function.shape[0])], axis=0)

        return - self.alpha * dff + function

    def _cauchy_navier_matrix(self, input_shape):
        assert isinstance(input_shape, tuple)
        len_input_shape = len(input_shape)
        assert len_input_shape > 0
        assert all([i > 0 for i in input_shape])

        def recursive_kronecker_product(dim, i=0):
            assert 0 <= dim <= len_input_shape - 1
            assert len_input_shape == 1 or 0 <= i <= len_input_shape - 2

            if len_input_shape == 1:
                diag = np.ones(input_shape[dim])
                laplacian = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1],
                                       input_shape[dim], input_shape[dim])
                return laplacian
            if i == len_input_shape - 2:
                if i == dim:
                    diag = np.ones(input_shape[dim])
                    laplacian = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1],
                                           input_shape[dim], input_shape[dim])
                    return sp.kron(laplacian, sp.eye(input_shape[i+1]))
                if dim == len_input_shape - 1:
                    diag = np.ones(input_shape[dim])
                    laplacian = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1],
                                           input_shape[dim], input_shape[dim])
                    return sp.kron(sp.eye(input_shape[i]), laplacian)
                return sp.kron(sp.eye(input_shape[i]), sp.eye(input_shape[i+1]))
            if i == dim:
                diag = np.ones(input_shape[dim])
                laplacian = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1],
                                       input_shape[dim], input_shape[dim])
                return sp.kron(laplacian, recursive_kronecker_product(dim, i+1))
            return sp.kron(sp.eye(input_shape[i]),
                           recursive_kronecker_product(dim, i+1))

        size = tuple_product(input_shape)
        mat = sp.csc_matrix((size, size))
        for dimension in range(len_input_shape):
            mat += recursive_kronecker_product(dimension)
        return (- self.alpha * mat + sp.eye(size))**self.exponent

    def cauchy_navier_squared_inverse(self, function):
        """Application of the operator `K=(LL)^-1` where `L` is the Cauchy-Navier type operator.

        Due to the structure of the operator it is easier to apply the operator in Fourier space.

        Parameters
        ----------
        function
            Array that holds the function values.

        Returns
        -------
        Array of the same shape as the input.
        """
        # check if helper operator is already defined
        if self.helper_operator is None or self.helper_operator.shape != function.shape[1:]:
            self.helper_operator = self.compute_helper_operator(function.shape)

        # transform input to Fourier space
        function_fourier = fftn(function)

        # perform operation in Fourier space
        result_fourier = function_fourier / self.helper_operator**2

        # transform back
        return ifftn(result_fourier)

    def compute_helper_operator(self, shape):
        """Computes the helper operator for the inverse of the squared Cauchy-Navier type operator.

        Parameters
        ----------
        shape
            Tuple containing the shape of the input image.

        Returns
        -------
        Operator as array.
        """
        dim = shape[0]
        shape = shape[1:]
        helper_operator = np.ones(shape, dtype=np.double)

        for i in np.ndindex(shape):
            for d in range(dim):
                helper_operator[i] += 2 * self.alpha * (1 - np.cos(2 * np.pi * i[d] / shape[d]))

        return np.stack([helper_operator, ] * dim, axis=0)

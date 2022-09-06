import numpy as np
from scipy.ndimage import convolve

from geodesic_shooting.core import VectorField
from geodesic_shooting.utils.helper_functions import tuple_product
from geodesic_shooting.utils.logger import getLogger


class BiharmonicRegularizer:
    """Biharmonic regularizer implementing smoothing functions.

    This class implements a regularizer for vector fields to make them smooth,
    such that the corresponding flows define diffeomorphisms.
    """
    def __init__(self, alpha=1, exponent=1, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        alpha
            Smoothness parameter that determines how strong the smoothing effect should be.
        exponent
            Penalty weight that ensures that the operator is non-singular.
        """
        assert alpha >= 0
        assert exponent > 0
        assert isinstance(exponent, int)

        self.alpha = alpha
        self.exponent = exponent

        self.helper_operator = None

        self.cauchy_navier_matrix = None
        self.cauchy_navier_inverse_matrix = None

        self.logger = getLogger('reduced_geodesic_shooting', level=log_level)

    def init_matrices(self, shape):
        """Initializes the Cauchy-Navier operator matrix and inverse matrices.
        It is very time-consuming to compute the inverse, but solving many linear
        systems of equations in each iteration is costly as well.
        Parameters
        ----------
        shape
            Shape of the input images.
        """
        dim = len(shape)
        self.cauchy_navier_matrix = np.kron(np.eye(dim, dtype=int),
                                            self._cauchy_navier_matrix(shape))
        self.logger.warning("Computing inverse of Cauchy-Navier operator matrix ...")
        inv_matrix = np.linalg.inv(self._cauchy_navier_matrix(shape))
        self.cauchy_navier_inverse_matrix = np.kron(np.eye(dim, dtype=int), inv_matrix)
        self.logger.info("Finished initialization of regularization matrices ...")

    def cauchy_navier(self, v):
        """Application of the Cauchy-Navier type operator (-alpha * Δ + exponent * I) to a function.

        Parameters
        ----------
        v
            `VectorField` to apply the operator to.

        Returns
        -------
        `VectorField` of the same shape as the input.
        """
        assert isinstance(v, VectorField)
        # check if helper operator is already defined
        if self.helper_operator is None or self.helper_operator.shape != v.spatial_shape:
            self.helper_operator = self.compute_helper_operator(v.dim, v.spatial_shape)

        # transform input to Fourier space
        function_fourier = self.fftn(v.to_numpy())

        # perform operation in Fourier space
        result_fourier = function_fourier * self.helper_operator

        # transform back
        result_inverse_fourier = self.ifftn(result_fourier)

        return VectorField(spatial_shape=v.spatial_shape, data=result_inverse_fourier)

    def _cauchy_navier_matrix(self, input_shape):
        assert isinstance(input_shape, tuple)
        len_input_shape = len(input_shape)
        assert len_input_shape > 0
        assert all([i > 0 for i in input_shape])

        def recursive_kronecker_product(dim, i=0):
            assert 0 <= dim <= len_input_shape - 1
            assert len_input_shape == 1 or 0 <= i <= len_input_shape - 2

            main_diagonal = -2. * np.ones(input_shape[dim])
            first_diagonal = np.ones(input_shape[dim]-1)
            laplacian = (np.diag(main_diagonal, 0) + np.diag(first_diagonal, 1)
                         + np.diag(first_diagonal, -1))

            if len_input_shape == 1:
                return laplacian
            if i == len_input_shape - 2:
                if i == dim:
                    return np.kron(laplacian, np.eye(input_shape[i+1]))
                if dim == len_input_shape - 1:
                    return np.kron(np.eye(input_shape[i]), laplacian)
                return np.kron(np.eye(input_shape[i]), np.eye(input_shape[i+1]))
            if i == dim:
                return np.kron(laplacian, recursive_kronecker_product(dim, i+1))
            return np.kron(np.eye(input_shape[i]),
                           recursive_kronecker_product(dim, i+1))

        size = tuple_product(input_shape)
        mat = np.zeros((size, size))
        for dimension in range(len_input_shape):
            mat += recursive_kronecker_product(dimension)
        return np.linalg.matrix_power((- self.alpha * mat + np.eye(size)), self.exponent)

    def cauchy_navier_inverse(self, v):
        """Application of the operator `K=L^{-1}` where `L` is the Cauchy-Navier type operator.

        Due to the structure of the operator it is easier to apply the operator in Fourier space.

        Parameters
        ----------
        v
            `VectorField` to apply the inverse operator to.

        Returns
        -------
        `VectorField` of the same shape as the input.
        """
        assert isinstance(v, VectorField)
        # check if helper operator is already defined
        if self.helper_operator is None or self.helper_operator.shape != v.spatial_shape:
            self.helper_operator = self.compute_helper_operator(v.dim, v.spatial_shape)

        # transform input to Fourier space
        function_fourier = self.fftn(v.to_numpy())

        # perform operation in Fourier space
        result_fourier = function_fourier / self.helper_operator

        # transform back
        result_inverse_fourier = self.ifftn(result_fourier)

        return VectorField(spatial_shape=v.spatial_shape, data=result_inverse_fourier)

    def compute_helper_operator(self, dim, spatial_shape):
        """Computes the helper operator for the Cauchy-Navier type operator.

        Parameters
        ----------
        shape
            Tuple containing the shape of the input image.

        Returns
        -------
        Operator as array.
        """
        helper_operator = np.zeros(spatial_shape, dtype=np.double)

        for i in np.ndindex(spatial_shape):
            for d in range(dim):
                helper_operator[i] += 2.*self.alpha*(1. - np.cos(2.*np.pi*i[d]/spatial_shape[d]))

        helper_operator += 1.
        helper_operator = helper_operator**self.exponent

        return np.stack([helper_operator, ] * dim, axis=-1)

    def fftn(self, array):
        """Performs `n`-dimensional FFT along the first `n` axes of an `n+1`-dimensional array.

        Parameters
        ----------
        array
           Input array to perform FFT on.

        Returns
        -------
        Array of the same shape.
        """
        transformed_array = np.zeros(array.shape, dtype=np.complex128)
        for i in range(array.shape[-1]):
            transformed_array[..., i] = np.fft.fftn(array[..., i])
        return transformed_array

    def ifftn(self, array):
        """Performs `n`-dimensional iFFT along the first `n` axes of an `n+1`-dimensional array.

        Parameters
        ----------
        array
           Input array to perform iFFT on.

        Returns
        -------
        Array of the same shape.
        """
        inverse_transformed_array = np.zeros(array.shape, dtype=np.complex128)
        for i in range(array.shape[-1]):
            inverse_transformed_array[..., i] = np.fft.ifftn(array[..., i])
        # convert to real here since we assume that functions consist of real numbers!
        return np.real(inverse_transformed_array)

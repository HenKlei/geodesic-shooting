import numpy as np
import scipy.sparse as sps

from geodesic_shooting.core import VectorField
from geodesic_shooting.utils.helper_functions import tuple_product
from geodesic_shooting.utils.logger import getLogger


class BiharmonicRegularizer:
    """Biharmonic regularizer implementing smoothing functions.

    This class implements a regularizer for vector fields to make them smooth,
    such that the corresponding flows define diffeomorphisms.
    """
    def __init__(self, alpha=0.1, exponent=1, gamma=1., fourier=True, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        alpha
            Smoothness parameter that determines how strong the smoothing effect should be.
        exponent
        gamma
        fourier
            Determines whether to apply the regularizer in Fourier domain.
            If the regularizer is applied in Fourier domain, periodic boundary conditions
            are implicitly used. Otherwise, finite differences with homogeneous Dirichlet
            boundary conditions are applied.
        log_level
        """
        assert alpha >= 0
        assert exponent > 0
        assert gamma >= 0
        assert isinstance(exponent, int)

        self.alpha = alpha
        self.exponent = exponent
        self.gamma = gamma

        self.helper_operator = None

        self.helmholtz_matrix = None
        self.cauchy_navier_matrix = None

        self.fourier = fourier

        self.logger = getLogger('reduced_geodesic_shooting', level=log_level)

    def __str__(self):
        return f"{self.__class__.__name__}: alpha={self.alpha}, exponent={self.exponent}"

    def init_matrices(self, shape):
        """Initializes the Cauchy-Navier operator matrix and inverse matrices.

        Parameters
        ----------
        shape
            Shape of the input images.
        """
        self.logger.info("Initializing matrices for regularizer ...")
        prod = tuple_product(shape)
        self.helmholtz_matrix = self._helmholtz_matrix(shape)
        assert self.helmholtz_matrix.shape == (prod, prod)
        self.cauchy_navier_matrix = self.helmholtz_matrix.T @ self.helmholtz_matrix
        assert self.cauchy_navier_matrix.shape == (prod, prod)
        with self.logger.block("Computing LU decomposition of Cauchy Navier matrix ..."):
            self.lu_decomposed_cauchy_navier_matrix = sps.linalg.splu(self.cauchy_navier_matrix)
        self.logger.info("Done.")

    def helmholtz(self, v):
        """Application of the Helmholtz operator `L` to a vector field.

        Here, the (self-adjoint) Helmholtz operator `L` is given
        as `L = (-alpha * Δ + gamma * I)**exponent`.

        Parameters
        ----------
        v
            `VectorField` to apply the operator to.

        Returns
        -------
        `VectorField` of the same shape as the input.
        """
        assert isinstance(v, VectorField)
        if self.fourier:
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
        else:
            res = [self.helmholtz_matrix @ v.to_numpy()[..., d].flatten(order='F') for d in range(v.dim)]
            res = np.array(res).T.reshape(v.full_shape, order='F')
            return VectorField(spatial_shape=v.spatial_shape, data=res)

    def cauchy_navier(self, v):
        """Application of the Cauchy-Navier type operator `L^*L` to a vector field.

        Here, the (self-adjoint) Helmholtz operator `L` is given
        as `L = (-alpha * Δ + gamma * I)**exponent`.
        Due to the structure of the operator it is easier to apply the operator in Fourier space.
        However, this implicitly assumes periodic boundary conditions.

        Parameters
        ----------
        v
            `VectorField` to apply the operator to.

        Returns
        -------
        `VectorField` of the same shape as the input.
        """
        assert isinstance(v, VectorField)
        if self.fourier:
            # check if helper operator is already defined
            if self.helper_operator is None or self.helper_operator.shape != v.spatial_shape:
                self.helper_operator = self.compute_helper_operator(v.dim, v.spatial_shape)

            # transform input to Fourier space
            function_fourier = self.fftn(v.to_numpy())

            # perform operation in Fourier space
            result_fourier = function_fourier * self.helper_operator**2

            # transform back
            result_inverse_fourier = self.ifftn(result_fourier)

            return VectorField(spatial_shape=v.spatial_shape, data=result_inverse_fourier)
        else:
            res = [self.cauchy_navier_matrix @ v.to_numpy()[..., d].flatten(order='F') for d in range(v.dim)]
            res = np.array(res).T.reshape(v.full_shape, order='F')
            return VectorField(spatial_shape=v.spatial_shape, data=res)

    def _helmholtz_matrix(self, input_shape):
        assert isinstance(input_shape, tuple)
        len_input_shape = len(input_shape)
        assert len_input_shape > 0
        assert all([i > 0 for i in input_shape])

        def recursive_kronecker_product(dim, i=0):
            assert 0 <= dim <= len_input_shape - 1
            assert len_input_shape == 1 or 0 <= i <= len_input_shape - 2

            main_diagonal = -2. * np.ones(input_shape[dim]) * (input_shape[dim] - 1.)**2
            first_diagonal = np.ones(input_shape[dim]-1) * (input_shape[dim] - 1.)**2
            laplacian = sps.diags([main_diagonal, first_diagonal, first_diagonal], [0, 1, -1])

            if len_input_shape == 1:
                return laplacian
            if i == len_input_shape - 2:
                if i == dim:
                    return sps.kron(laplacian, sps.eye(input_shape[i+1]))
                if dim == len_input_shape - 1:
                    return sps.kron(sps.eye(input_shape[i]), laplacian)
                return sps.kron(sps.eye(input_shape[i]), sps.eye(input_shape[i+1]))
            if i == dim:
                return sps.kron(laplacian, recursive_kronecker_product(dim, i+1))
            return sps.kron(sps.eye(input_shape[i]),
                            recursive_kronecker_product(dim, i+1))

        size = tuple_product(input_shape)
        mat = np.zeros((size, size))
        for dimension in range(len_input_shape):
            mat += recursive_kronecker_product(dimension)
        return (- self.alpha * mat + self.gamma * sps.eye(size)) ** self.exponent

    def cauchy_navier_inverse(self, v):
        """Application of the operator `K=(L^*L)^{-1}`, with the Helmholtz operator `L`.

        Due to the structure of the operator it is easier to apply the inverse operator
        in Fourier space. However, this implicitly assumes periodic boundary conditions.

        Parameters
        ----------
        v
            `VectorField` to apply the inverse operator to.

        Returns
        -------
        `VectorField` of the same shape as the input.
        """
        assert isinstance(v, VectorField)
        if self.fourier:
            # check if helper operator is already defined
            if self.helper_operator is None or self.helper_operator.shape != v.spatial_shape:
                self.helper_operator = self.compute_helper_operator(v.dim, v.spatial_shape)

            # transform input to Fourier space
            function_fourier = self.fftn(v.to_numpy())

            # perform operation in Fourier space
            result_fourier = function_fourier / (self.helper_operator**2)

            # transform back
            result_inverse_fourier = self.ifftn(result_fourier)

            return VectorField(spatial_shape=v.spatial_shape, data=result_inverse_fourier)
        else:
            res_complete = []
            prod = tuple_product(v.spatial_shape)
            for d in range(v.dim):
                res = self.lu_decomposed_cauchy_navier_matrix.solve(v.to_numpy().flatten(order='F')[d*prod:(d+1)*prod])
                res_complete.append(res)
            res = np.array(res_complete).T.reshape(v.full_shape, order='F')
            return VectorField(spatial_shape=v.spatial_shape, data=res)

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
                helper_operator[i] += 2. * self.alpha * (1. - np.cos(2. * np.pi * i[d] / spatial_shape[d])) \
                                      * spatial_shape[d]**2

        helper_operator += self.gamma
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

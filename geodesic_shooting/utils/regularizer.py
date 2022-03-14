import numpy as np
from scipy.ndimage import convolve

from geodesic_shooting.core import VectorField


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
            Penalty weight that ensures that the operator is non-singular.
        """
        assert alpha > 0
        assert exponent > 0

        self.alpha = alpha
        self.exponent = exponent

        self.helper_operator = None

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
        dim = v.dim
        assert dim in [1, 2]

        # window depends on the dimension and represents a simple approximation
        # of the Laplace operator
        if dim == 1:
            window = np.array([1., -2., 1.])
        elif dim == 2:
            window = np.array([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]])

        dff = np.stack([convolve(v[..., d], window) for d in range(dim)], axis=-1)

        return self.exponent * v - self.alpha * dff

    def cauchy_navier_squared_inverse(self, v):
        """Application of the operator `K=(LL)^-1` where `L` is the Cauchy-Navier type operator.

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
        result_fourier = function_fourier / self.helper_operator**2

        # transform back
        result_inverse_fourier = self.ifftn(result_fourier)

        return VectorField(spatial_shape=v.spatial_shape, data=result_inverse_fourier)

    def compute_helper_operator(self, dim, spatial_shape):
        """Computes the helper operator for the inverse of the squared Cauchy-Navier type operator.

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
                helper_operator[i] += 2 * self.alpha * (1 - np.cos(2 * np.pi * i[d] / spatial_shape[d]))

        helper_operator += self.exponent

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
        # convert to real here since we assume that images consist of real numbers!
        return np.real(inverse_transformed_array)

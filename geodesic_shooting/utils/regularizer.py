import numpy as np
from scipy.ndimage import convolve


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

    def cauchy_navier(self, function):
        """Application of the Cauchy-Navier type operator (-alpha * Δ + exponent * I) to a function.

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

        return - self.alpha * dff + self.exponent * function

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
        function_fourier = self.fftn(function)

        # perform operation in Fourier space
        result_fourier = function_fourier / self.helper_operator**2

        # transform back
        return self.ifftn(result_fourier)

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
        helper_operator = np.zeros(shape, dtype=np.double)

        for i in np.ndindex(shape):
            for d in range(dim):
                helper_operator[i] += 2 * self.alpha * (1 - np.cos(2 * np.pi * i[d] / shape[d]))

        helper_operator += self.exponent

        return np.stack([helper_operator, ] * dim, axis=0)

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
        for i in range(array.shape[0]):
            transformed_array[i] = np.fft.fftn(array[i])
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
        for i in range(array.shape[0]):
            inverse_transformed_array[i] = np.fft.ifftn(array[i])
        # convert to real here since we assume that images consist of real numbers!
        return np.real(inverse_transformed_array)

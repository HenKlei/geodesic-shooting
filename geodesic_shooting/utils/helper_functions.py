import numpy as np


def tuple_product(val):
    res = 1
    for ele in val:
        res *= ele
    return res


def fftn(array):
    """Performs `n`-dimensional FFT along the last `n` axes of an `n+1`-dimensional array.

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


def ifftn(array):
    """Performs `n`-dimensional iFFT along the last `n` axes of an `n+1`-dimensional array.

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

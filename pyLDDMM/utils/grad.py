import numpy as np
from scipy.ndimage import convolve


def finite_difference(a):
    """
    Calculates the gradient of a via finite differences
    @param a: an array
    @return: array, with partial derivatives
    """
    w = np.array([1., 0., -1.])  # unusual ordering since we use convolution and not correlation
    dim = a.ndim
    w = w.reshape(list(w.shape)+[1,]*(dim-1)).T
    wy = w.T

    g = []

    for d in range(dim):
        indices = list(range(dim))
        indices[0] = d
        indices[d] = 0
        wd = np.transpose(w, axes=indices)
        gd = convolve(a, wd)
        g.append(gd)

    return np.flip(np.stack(g, axis=0), axis=0)


if __name__ == "__main__":
    img = np.zeros((5, 10))
    img[..., 2] = 1
    derivative = np.zeros((2, 5, 10))
    derivative[1, :, 1] = 1
    derivative[1, :, 3] = -1
    assert (finite_difference(img) == derivative).all()

    img = np.zeros((5, 10))
    img[2, ...] = 1
    derivative = np.zeros((2, 5, 10))
    derivative[0, 1, :] = 1
    derivative[0, 3, :] = -1
    assert (finite_difference(img) == derivative).all()

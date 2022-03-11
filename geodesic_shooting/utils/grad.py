import numpy as np
from scipy.ndimage import correlate

import geodesic_shooting.core as core


def finite_difference(f):
    """Finite difference scheme for approximating the derivative of the input array.

    This function uses central differences to compute the (discrete) derivative
    of the input array in the different dimensions.

    Parameters
    ----------
    array
        Input array to compute the derivative of.

    Returns
    -------
    Array containing the derivatives in the different dimensions.
    """
    assert isinstance(f, core.VectorField) or isinstance(f, core.ScalarFunction)

    window = np.array([-1., 0., 1.]) * 0.5
    dim = f.dim
    window = window.reshape(list(window.shape) + [1, ]*(dim-1))

    def _fd_single_dim(u, d):
        indices = list(range(dim))
        indices[0] = d
        indices[d] = 0
        window_d = np.transpose(window, axes=indices)
        return correlate(u, window_d)

    derivatives = []

    if isinstance(f, core.VectorField):
        for i in range(f.dim):
            derivatives_d = []
            for j in range(f.dim):
                derivatives_d.append(_fd_single_dim(f[..., i], j))
            derivatives.append(np.stack(derivatives_d, axis=-1))
        return np.stack(derivatives, axis=-1)

    if isinstance(f, core.ScalarFunction):
        for d in range(dim):
            derivatives.append(_fd_single_dim(f.to_numpy(), d))
        return core.VectorField(spatial_shape=f.spatial_shape, data=np.stack(derivatives, axis=-1))

    raise NotImplementedError

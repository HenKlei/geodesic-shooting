import numpy as np
from scipy.ndimage import correlate

import geodesic_shooting.core as core


def finite_difference(f):
    """Finite difference scheme for approximating the derivative of the input.

    This function uses central differences to compute the (discrete) derivative
    of the input in the different dimensions.

    Parameters
    ----------
    f
        `ScalarFunction` or `VectorField` to compute the derivative of.

    Returns
    -------
    In the case of a `ScalarFunction` `f`, the result is a gradient `VectorField`,
    in the case of a `VectorField` `f`, the result is a numpy-array containing the
    gradient/Jacobian of the `VectorField` at the spatial points.
    """
    assert isinstance(f, (core.ScalarFunction, core.VectorField))

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

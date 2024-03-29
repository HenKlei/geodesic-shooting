import numpy as np

import geodesic_shooting.core as core


def coordinate_grid(shape):
    """Function for generating a coordinate grid that corresponds to the identity mapping.

    Parameters
    ----------
    shape
        Spatial shape of the coordinate grid.

    Returns
    -------
    The `VectorField` with spatial_shape `shape`.
    """
    assert len(shape) > 0

    if len(shape) == 1:
        return core.VectorField(spatial_shape=shape, data=np.mgrid[:shape[0]][..., np.newaxis].astype(np.double))

    ranges = [np.arange(s) for s in shape]
    return core.VectorField(spatial_shape=shape,
                            data=np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).astype(np.double))


def identity_diffeomorphism(shape):
    """Function for generating a coordinate grid that corresponds to the identity mapping.

    Parameters
    ----------
    shape
        Spatial shape of the coordinate grid.

    Returns
    -------
    The `VectorField` with spatial_shape `shape`.
    """
    assert len(shape) > 0

    if len(shape) == 1:
        return core.Diffeomorphism(spatial_shape=shape, data=np.mgrid[:shape[0]][..., np.newaxis].astype(np.double))

    ranges = [np.arange(s) for s in shape]
    return core.Diffeomorphism(spatial_shape=shape,
                               data=np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).astype(np.double))

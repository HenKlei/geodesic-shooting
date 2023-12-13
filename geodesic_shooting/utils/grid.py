import numpy as np

import torch

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

    ranges = [np.arange(s) for s in shape]
    return core.VectorField(spatial_shape=shape,
                            data=torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1).astype(np.double))


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

    ranges = [np.arange(s) for s in shape]
    return core.Diffeomorphism(spatial_shape=shape,
                               data=torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=-1).astype(np.double))

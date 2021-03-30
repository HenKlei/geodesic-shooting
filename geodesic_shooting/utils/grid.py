import numpy as np


def coordinate_grid(shape):
    """Function for generating a coordinate grid that corresponds to the identity mapping.

    Parameters
    ----------
    shape
        Spatial shape of the coordinate grid.

    Returns
    -------
    The coordinate grid of size (dim x shape).
    """
    assert len(shape) > 0

    if len(shape) == 1:
        return np.mgrid[:shape[0]][np.newaxis, ...]

    return np.mgrid[[slice(0, s) for s in shape]]

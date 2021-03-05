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


if __name__ == "__main__":
    grid_shape = (4,)
    grid = coordinate_grid(grid_shape)
    assert grid.shape == (1,) + grid_shape

    grid_shape = (4, 3)
    grid = coordinate_grid(grid_shape)
    assert grid.shape == (2,) + grid_shape

    grid_shape = (4, 3, 5)
    grid = coordinate_grid(grid_shape)
    assert grid.shape == (3,) + grid_shape

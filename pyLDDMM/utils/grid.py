import numpy as np


def coordinate_grid(shape):
    """
    generates a coordinate grid of dimension shape[0] x shape[1] x 2
    @param shape: tuple
    @return: grid
    """
    assert len(shape) in [1, 2]

    if len(shape) == 1:
        return np.transpose(np.array([np.mgrid[slice(0, shape[0], 1)],], dtype=np.double))

    grid = np.mgrid[:shape[0], :shape[1]]
    return np.transpose(grid, axes=[2, 1, 0])

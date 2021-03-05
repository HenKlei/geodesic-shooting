import numpy as np


def coordinate_grid(shape):
    """
    generates a coordinate grid of dimension shape[0] x shape[1] x 2
    @param shape: tuple
    @return: grid
    """
    assert len(shape) in [1, 2]

    if len(shape) == 1:
        return np.mgrid[:shape[0]][np.newaxis, ...]

    return np.mgrid[:shape[0], :shape[1]]


if __name__ == "__main__":
    shape = (4, 3)
    grid = coordinate_grid(shape)
    assert grid.shape == (2,) + shape

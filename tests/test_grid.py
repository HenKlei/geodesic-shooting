from geodesic_shooting.utils.grid import coordinate_grid


def test_grid():
    grid_shape = (4,)
    grid = coordinate_grid(grid_shape)
    assert grid.shape == (1,) + grid_shape

    grid_shape = (4, 3)
    grid = coordinate_grid(grid_shape)
    assert grid.shape == (2,) + grid_shape

    grid_shape = (4, 3, 5)
    grid = coordinate_grid(grid_shape)
    assert grid.shape == (3,) + grid_shape

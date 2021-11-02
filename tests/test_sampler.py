import numpy as np

from geodesic_shooting.utils import grid, sampler


def test_sampler():
    shape = (5, 10)
    input_coordinates = grid.coordinate_grid(shape)
    array1 = np.random.rand(*shape)

    # single color channel...
    result1 = sampler.sample(array1, input_coordinates)
    assert (array1 == result1).all()

    array2 = np.random.rand(*shape)
    input_array = np.stack([array1, array2], axis=0)

    # two color channels...
    result = sampler.sample(input_array, input_coordinates)
    assert (input_array == result).all()

import numpy as np

from geodesic_shooting.utils import grid, sampler
from geodesic_shooting.core import ScalarFunction, VectorField


def test_sampler():
    shape = (5, 10)
    input_coordinates = grid.coordinate_grid(shape)
    array1 = ScalarFunction(shape, data=np.random.rand(*shape))

    # single color channel...
    result1 = sampler.sample(array1, input_coordinates)
    assert array1 == result1

    array2 = np.random.rand(*shape)
    input_array = VectorField(shape, data=np.stack([array1.to_numpy(), array2], axis=-1))

    # two color channels...
    result = sampler.sample(input_array, input_coordinates)
    assert input_array == result

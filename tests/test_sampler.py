import numpy as np

from geodesic_shooting.utils import grid, sampler
from geodesic_shooting.core import ScalarFunction, VectorField


def test_sampler():
    shape = (5, 10)
    input_coordinates = grid.identity_diffeomorphism(shape)
    array1 = ScalarFunction(shape, data=np.random.rand(*shape))

    result1 = sampler.sample(array1, input_coordinates)
    assert array1 == result1

    array2 = np.random.rand(*shape)
    input_array = VectorField(shape, data=np.stack([array1.to_numpy(), array2], axis=-1))

    result = sampler.sample(input_array, input_coordinates)
    assert input_array == result


def test_sampler_inverse():
    shape = (10, 10)
    data_u0 = np.zeros(shape)
    data_u0[5, 2] = 1.
    data_u0[0, 4] = 2.
    u0 = ScalarFunction(data=data_u0)

    v0 = grid.identity_diffeomorphism(shape)
    uniform_diff_vector = np.array([3, 4])
    v0[:, :] -= uniform_diff_vector
    u1 = u0.push_forward(v0)

    back_u1 = sampler.sample_inverse(u1, v0)

    assert np.isclose((back_u1 - u0).norm, 0.)

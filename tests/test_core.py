import numpy as np

from geodesic_shooting.core import ScalarFunction, VectorField
from geodesic_shooting.utils.helper_functions import tuple_product


def test_functions():
    shape = (5, 10)
    f = ScalarFunction(shape)

    assert isinstance(f.to_numpy(), np.ndarray)
    assert f.spatial_shape == f.full_shape == shape
    assert f.size == tuple_product(shape) == 50
    assert np.isclose(f.norm, 0.)

    f[0, 0] = 10.
    assert np.isclose(f.norm, 10.)
    assert np.isclose(f.get_norm(order=np.inf), 10.)
    assert np.isclose(f.get_norm(order=5), 10.)

    g = f.copy()
    assert np.isclose((f - g).norm, 0.)
    assert np.isclose((g - f).norm, 0.)
    assert np.isclose((f + g).norm, 20.)
    assert np.isclose((g + f).norm, 20.)
    assert np.isclose((2. * f + g).norm, 30.)


def test_vector_fields():
    shape = (5, 10)
    v = VectorField(shape)

    assert isinstance(v.to_numpy(), np.ndarray)
    assert v.spatial_shape == shape
    assert v.full_shape == (5, 10, 2)
    assert np.isclose(v.norm, 0.)

    v[0, 0, 0] = 10.
    assert np.isclose(v.norm, 10.)
    assert np.isclose(v.get_norm(order=np.inf), 10.)
    assert np.isclose(v.get_norm(order=5), 10.)

    w = v.copy()
    assert np.isclose((v - w).norm, 0.)
    assert np.isclose((w - v).norm, 0.)
    assert np.isclose((v + w).norm, 20.)
    assert np.isclose((w + v).norm, 20.)
    assert np.isclose((2. * v + w).norm, 30.)

import numpy as np

from geodesic_shooting.utils.grad import finite_difference
from geodesic_shooting.core import ScalarFunction, VectorField


def test_grad():
    f1 = ScalarFunction((5, 10))
    f1[..., 2] = 1
    derivative = VectorField((5, 10))
    derivative[:, 1, 1] = 0.5
    derivative[:, 3, 1] = -0.5
    assert finite_difference(f1) == derivative

    f2 = ScalarFunction((5, 10))
    f2[2, ...] = 1
    derivative = VectorField((5, 10))
    derivative[1, :, 0] = 0.5
    derivative[3, :, 0] = -0.5
    assert finite_difference(f2) == derivative

    v = VectorField((5, 10))
    v[..., 0] = f1
    v[..., 1] = f2
    assert (finite_difference(v) == np.stack([finite_difference(f1).to_numpy(),
                                              finite_difference(f2).to_numpy()], axis=-1)).all()

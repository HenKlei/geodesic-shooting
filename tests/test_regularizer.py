import numpy as np

from geodesic_shooting.utils.regularizer import BiharmonicRegularizer
from geodesic_shooting.core import VectorField


def test_regularizer_self_adjoint():
    regularizer = BiharmonicRegularizer(alpha=1, exponent=2)
    v = VectorField((6, 4))
    v[2, 2, 0] = 1.
    w = VectorField(v.spatial_shape)
    w[3, 2, 0] = 2.

    wLv = w.to_numpy().flatten().dot(regularizer.cauchy_navier(v).to_numpy().flatten())
    vLw = v.to_numpy().flatten().dot(regularizer.cauchy_navier(w).to_numpy().flatten())
    assert np.isclose(wLv, vLw)

import numpy as np
import pytest

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


@pytest.mark.parametrize("alpha", [1., 0.1, 0.01])
@pytest.mark.parametrize("exponent", [1])
@pytest.mark.parametrize("gamma", [10., 1., 0.1])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("size_per_dimension", [10, 20])
def test_regularizer_matrices(alpha, exponent, gamma, dim, size_per_dimension):
    regularizer = BiharmonicRegularizer(alpha=alpha, exponent=exponent, gamma=gamma, fourier=False)
    regularizer.init_matrices(tuple([size_per_dimension] * dim))

    assert np.allclose(regularizer.helmholtz_matrix, regularizer.helmholtz_matrix.T)
    assert regularizer.helmholtz_matrix.shape == (size_per_dimension**dim * dim, size_per_dimension**dim * dim)
    if dim == 1:
        assert np.isclose(regularizer.helmholtz_matrix[0, 0], gamma + alpha * 2. * (size_per_dimension - 1) ** 2)
        assert np.isclose(regularizer.helmholtz_matrix[0, 1], -alpha * (size_per_dimension - 1) ** 2)
        assert np.isclose(regularizer.helmholtz_matrix[1, 0], -alpha * (size_per_dimension - 1) ** 2)
        assert np.allclose(regularizer.helmholtz_matrix,
                           np.diag((gamma + alpha * 2. * (size_per_dimension - 1) ** 2) * np.ones(size_per_dimension))
                           + np.diag(-alpha * (size_per_dimension - 1) ** 2 * np.ones(size_per_dimension - 1), 1)
                           + np.diag(-alpha * (size_per_dimension - 1) ** 2 * np.ones(size_per_dimension - 1), -1))
    elif dim == 2:
        assert np.allclose(regularizer.helmholtz_matrix[:size_per_dimension**2, :size_per_dimension**2],
                           regularizer.helmholtz_matrix[size_per_dimension**2:, size_per_dimension**2:])
        assert np.allclose(regularizer.helmholtz_matrix[:size_per_dimension**2, size_per_dimension**2:],
                           np.zeros((size_per_dimension**2, size_per_dimension**2)))
        assert np.allclose(regularizer.helmholtz_matrix[size_per_dimension**2:, :size_per_dimension**2],
                           np.zeros((size_per_dimension**2, size_per_dimension**2)))
        assert np.isclose(regularizer.helmholtz_matrix[0, 0], gamma * 1 + alpha * 4. * (size_per_dimension - 1) ** 2)

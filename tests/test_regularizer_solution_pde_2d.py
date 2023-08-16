import numpy as np
import pytest
import scipy.sparse as sps

from geodesic_shooting.core import VectorField
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


@pytest.mark.parametrize("N", [10, 50])
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("exponent", [1])
@pytest.mark.parametrize("gamma", [1., 5., 10.])
def test_regularizer_as_PDE_solver_2d(N, alpha, exponent, gamma):
    x_vals = np.linspace(0, 1, N)
    assert len(x_vals) == N

    b = np.ones(N**2)

    regularizer = BiharmonicRegularizer(alpha=alpha, exponent=exponent, gamma=gamma, fourier=False)
    regularizer.init_matrices((N, N))

    A2 = regularizer.cauchy_navier_matrix
    u, info = sps.linalg.cg(A2, b)
    assert info == 0
    u2 = regularizer.lu_decomposed_cauchy_navier_matrix.solve(b)
    assert np.allclose(u, u2)

    u_square = np.reshape(u, [N, N])

    data = np.zeros((N, N, 2))
    data[..., 0] = 1.
    data[..., 1] = 2.
    v = VectorField(data=data)
    u = regularizer.cauchy_navier_inverse(v)
    assert np.allclose(u_square, u.to_numpy()[..., 0])
    assert np.allclose(2. * u_square, u.to_numpy()[..., 1])

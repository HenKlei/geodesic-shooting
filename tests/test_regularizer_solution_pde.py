import numpy as np
import pytest
import scipy.sparse as sps

from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


@pytest.mark.parametrize("N", [10, 100, 1000])
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("exponent", [1])
@pytest.mark.parametrize("gamma", [5., 10.])
def test_regularizer_as_PDE_solver_1d(N, alpha, exponent, gamma):
    x_vals = np.linspace(0, 1, N)
    assert len(x_vals) == N

    A = (np.diag((gamma + alpha * 2. * (N - 1) ** 2) * np.ones(N))
         + np.diag(-alpha * (N - 1) ** 2 * np.ones(N - 1), 1)
         + np.diag(-alpha * (N - 1) ** 2 * np.ones(N - 1), -1))
    A = np.linalg.matrix_power(A, exponent)
    b = x_vals

    y_vals = np.linalg.solve(A, b)

    regularizer = BiharmonicRegularizer(alpha=alpha, exponent=exponent, gamma=gamma, fourier=False, spatial_shape=(N, ))

    A2 = regularizer.helmholtz_matrix
    y_vals_2 = sps.linalg.spsolve(A2, b)
    assert np.allclose(y_vals, y_vals_2)
    assert np.linalg.norm(A2 @ y_vals_2 - b) < 1e-9

    A3 = regularizer.cauchy_navier_matrix
    y_vals_3 = sps.linalg.spsolve(A3, b)
    y_vals_check = np.linalg.solve(A, y_vals)
    assert np.allclose(y_vals_3, y_vals_check)
    assert np.linalg.norm(A3 @ y_vals_3 - b) < 5e-5

    y_vals_4 = regularizer.lu_decomposed_cauchy_navier_matrix.solve(b)
    assert np.allclose(y_vals_4, y_vals_check)
    assert np.linalg.norm(A3 @ y_vals_4 - b) < 5e-5

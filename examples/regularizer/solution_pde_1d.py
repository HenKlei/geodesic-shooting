import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


N = 1000
x_vals = np.linspace(0, 1, N)
assert len(x_vals) == N

alpha = 0.5
exponent = 1
gamma = 10.

A = (np.diag((gamma + alpha * 2. * (N - 1) ** 2) * np.ones(N))
     + np.diag(-alpha * (N - 1) ** 2 * np.ones(N - 1), 1)
     + np.diag(-alpha * (N - 1) ** 2 * np.ones(N - 1), -1))
b = x_vals

y_vals = np.linalg.solve(A, b)
plt.plot(x_vals, y_vals)
plt.grid()
plt.show()

regularizer = BiharmonicRegularizer(alpha=alpha, exponent=exponent, gamma=gamma, fourier=False)
regularizer.init_matrices((N, ))

A2 = regularizer.helmholtz_matrix
y_vals_2 = sps.linalg.spsolve(A2, b)
assert np.allclose(y_vals, y_vals_2)
assert np.linalg.norm(A2 @ y_vals_2 - b) < 1e-10
plt.plot(x_vals, y_vals_2)
plt.grid()
plt.show()

A3 = regularizer.cauchy_navier_matrix
y_vals_3 = sps.linalg.spsolve(A3, b)
y_vals_check = np.linalg.solve(A, y_vals)
assert np.allclose(y_vals_3, y_vals_check)
assert np.linalg.norm(A3 @ y_vals_3 - b) < 1e-5
plt.plot(x_vals, y_vals_3)
plt.grid()
plt.show()

y_vals_4 = regularizer.lu_decomposed_cauchy_navier_matrix.solve(b)
assert np.allclose(y_vals_4, y_vals_check)
assert np.linalg.norm(A3 @ y_vals_4 - b) < 1e-5
plt.plot(x_vals, y_vals_4)
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import time

from geodesic_shooting.core import VectorField
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


N = 50
x_vals = np.linspace(0, 1, N)
assert len(x_vals) == N

alpha = 0.5
exponent = 1
gamma = 1.

b = np.ones(N**2)

regularizer = BiharmonicRegularizer(alpha=alpha, exponent=exponent, gamma=gamma, fourier=False)
regularizer.init_matrices((N, N))

A2 = regularizer.cauchy_navier_matrix

tic = time.perf_counter()
u, info = sps.linalg.cg(A2, b)
print(f"Time CG: {time.perf_counter() - tic}")
assert info == 0
tic = time.perf_counter()
u2 = regularizer.lu_decomposed_cauchy_navier_matrix.solve(b)
print(f"Time with precomputed LU: {time.perf_counter() - tic}")
assert np.allclose(u, u2)

u_square = np.reshape(u, [N, N])

levels = None
plt.contourf(u_square, levels=levels)
plt.colorbar()
plt.show()

data = np.zeros((N, N, 2))
data[..., 0] = 1.
data[..., 1] = 2.
v = VectorField(data=data)
v.plot()
u = regularizer.cauchy_navier_inverse(v)

levels = None
plt.contourf(u.to_numpy()[..., 0], levels=levels)
plt.colorbar()
plt.show()
assert np.allclose(u_square, u.to_numpy()[..., 0])

plt.contourf(u.to_numpy()[..., 1], levels=levels)
plt.colorbar()
plt.show()
assert np.allclose(2. * u_square, u.to_numpy()[..., 1])

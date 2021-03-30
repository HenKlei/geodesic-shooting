import numpy as np

from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


v = np.zeros((2, 5, 5))
v[0, 2, 2] = 1

regularizer = BiharmonicRegularizer(alpha=1, exponent=1)

print(regularizer.cauchy_navier_squared_inverse(v)[0])

import numpy as np

from geodesic_shooting.utils.regularizer import BiharmonicRegularizer
from geodesic_shooting.core import VectorField


v = VectorField((5, 5))
v[2, 2, 0] = 1

regularizer = BiharmonicRegularizer(alpha=1, exponent=1)

print(regularizer.cauchy_navier_squared_inverse(v)[0])

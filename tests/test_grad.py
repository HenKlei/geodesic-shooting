import numpy as np

from geodesic_shooting.utils.grad import finite_difference


img = np.zeros((5, 10))
img[..., 2] = 1
derivative = np.zeros((2, 5, 10))
derivative[1, :, 1] = 1
derivative[1, :, 3] = -1
assert (finite_difference(img) == derivative).all()

img = np.zeros((5, 10))
img[2, ...] = 1
derivative = np.zeros((2, 5, 10))
derivative[0, 1, :] = 1
derivative[0, 3, :] = -1
assert (finite_difference(img) == derivative).all()

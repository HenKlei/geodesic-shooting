import matplotlib.pyplot as plt
import numpy as np

import geodesic_shooting
from geodesic_shooting.core import VectorField


if __name__ == "__main__":
    n = 20
    shape = (n, n)

    def f(x, y):
        return np.stack([2.*np.ones_like(x), 3.*np.ones_like(x)], axis=-1)

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing='ij')

    displacement_field = VectorField(data=f(grid_x, grid_y))
    time_steps = 5

    gs = geodesic_shooting.GeodesicShooting(time_steps=time_steps)
    time_dependent_field = gs.integrate_forward_vector_field(displacement_field)
    assert all([np.isclose((time_dependent_field[t] - displacement_field).norm, 0.) for t in range(time_steps)])
    _ = time_dependent_field.animate()
    plt.show()

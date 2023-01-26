import matplotlib.pyplot as plt
import numpy as np

from geodesic_shooting.core import VectorField


if __name__ == "__main__":
    n = 20
    shape = (n, n)
    identity_grid = VectorField(data=np.stack(np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float)),
                                              axis=-1))
    identity_grid.plot_as_warpgrid(title="Identity grid")

    def f(x, y):
        return np.stack([0.8*np.exp(-x**2-y**2), -0.4*np.exp(-x**2-y**2)], axis=-1)

    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, shape[0]), np.linspace(-3, 3, shape[1]))
    displacement_field = VectorField(data=f(grid_x, grid_y) * n / 6)
    vector_field = identity_grid + displacement_field
    vector_field.plot_as_warpgrid(title="Gaussian deformation")
    vector_field.plot_as_warpgrid(title="Gaussian deformation without identity grid", show_identity_grid=False)
    vector_field.plot_as_warpgrid(title="Gaussian deformation with displacement vectors",
                                  show_displacement_vectors=True)
    vector_field.plot_as_warpgrid(title="Gaussian deformation with displacement vectors and without identity grid",
                                  show_identity_grid=False, show_displacement_vectors=True)
    plt.show()

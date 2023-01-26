import matplotlib.pyplot as plt
import numpy as np

from geodesic_shooting.core import VectorField


if __name__ == "__main__":
    n = 20
    shape = (n, 2*n)
    zero_vector_field = VectorField(spatial_shape=shape)
    zero_vector_field.plot_as_warpgrid(title="Identity grid")

    def f(x, y):
        return np.stack([0.8*np.exp(-x**2-y**2) * shape[0] / 6, -0.4*np.exp(-x**2-y**2) * shape[1] / 6], axis=-1)

    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, shape[0]), np.linspace(-3, 3, shape[1]), indexing='ij')
    displacement_field = VectorField(data=f(grid_x, grid_y))
    displacement_field.plot_as_warpgrid(title="Gaussian deformation")
    displacement_field.plot_as_warpgrid(title="Gaussian deformation without identity grid", show_identity_grid=False)
    displacement_field.plot_as_warpgrid(title="Gaussian deformation with displacement vectors",
                                        show_displacement_vectors=True)
    displacement_field.plot_as_warpgrid(title="Gaussian deformation with displacement vectors, without identity grid",
                                        show_identity_grid=False, show_displacement_vectors=True)
    plt.show()

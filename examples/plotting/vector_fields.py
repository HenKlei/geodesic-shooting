import matplotlib.pyplot as plt
import numpy as np

from geodesic_shooting.core import VectorField


if __name__ == "__main__":
    n = 20
    shape = (n, n)

    def f(x, y):
        return np.stack([0.75*(1 - x) / np.exp(np.sqrt((1 - x)**2 + y**2))
                         + 100*(10 - x) / np.exp(np.sqrt((10 - x)**2 + y**2)),
                         -y / np.exp(np.sqrt((1 - x)**2 + y**2))],
                        axis=-1)

    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, shape[0]), np.linspace(-3, 3, shape[1]))
    displacement_field = VectorField(data=f(grid_x, grid_y) * n / 6)
    identity_grid = VectorField(data=np.stack(np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float)),
                                              axis=-1))
    vector_field = displacement_field + identity_grid
    vector_field.plot_as_warpgrid(title="Warpgrid", show_displacement_vectors=True)
    displacement_field.plot(title="Displacement vector field")
    displacement_field.plot(title="Displacement vector field with colors", color_length=True)
    displacement_field.plot_streamlines(title="Streamlines")
    displacement_field.plot_streamlines(title="Streamlines with colors", color_length=True, density=2)
    plt.show()

    def f(x, y):
        return np.stack([0.8*np.exp(-x**2-y**2), -0.4*np.exp(-x**2-y**2)], axis=-1)

    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, shape[0]), np.linspace(-3, 3, shape[1]))
    displacement_field = VectorField(data=f(grid_x, grid_y) * n / 6)
    vector_field = identity_grid + displacement_field
    vector_field.plot_as_warpgrid(title="Gaussian deformation")
    displacement_field.plot(title="Displacement vector field")
    displacement_field.plot(title="Displacement vector field with colors", color_length=True)
    displacement_field.plot_streamlines(title="Streamlines")
    displacement_field.plot_streamlines(title="Streamlines with colors", color_length=True, density=2)
    plt.show()

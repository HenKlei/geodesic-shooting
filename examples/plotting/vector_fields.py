import matplotlib.pyplot as plt
import numpy as np

from geodesic_shooting.core import VectorField


if __name__ == "__main__":
    n = 20
    shape = (n, 2*n)
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, shape[0]), np.linspace(-3, 3, shape[1]), indexing='ij')

    def f(x, y):
        return np.stack([(0.75*(1 - x) / np.exp(np.sqrt((1 - x)**2 + y**2))
                          + 100*(10 - x) / np.exp(np.sqrt((10 - x)**2 + y**2))) / 6,
                         -y / np.exp(np.sqrt((1 - x)**2 + y**2)) / 6],
                        axis=-1)

    displacement_field = VectorField(data=f(grid_x, grid_y))
    magnitude = displacement_field.get_magnitude()
    x_component = displacement_field.get_component_as_function(0)
    y_component = displacement_field.get_component_as_function(1)

    displacement_field.plot_as_warpgrid(title="Warpgrid", show_displacement_vectors=True)
    displacement_field.plot(title="Displacement vector field")
    displacement_field.plot(title="Displacement vector field with colors", color_length=True)
    displacement_field.plot_streamlines(title="Streamlines")
    displacement_field.plot_streamlines(title="Streamlines with colors", color_length=True, density=2)
    magnitude.plot(title="Magnitude of vector field")
    x_component.plot(title="x-component of vector field")
    y_component.plot(title="y-component of vector field")
    plt.show()

    def f(x, y):
        return np.stack([0.8*np.exp(-x**2-y**2) / 6, -0.4*np.exp(-x**2-y**2) / 6], axis=-1)

    displacement_field = VectorField(data=f(grid_x, grid_y))
    displacement_field.plot_as_warpgrid(title="Gaussian deformation")
    displacement_field.plot(title="Displacement vector field")
    displacement_field.plot(title="Displacement vector field with colors", color_length=True)
    displacement_field.plot_streamlines(title="Streamlines")
    displacement_field.plot_streamlines(title="Streamlines with colors", color_length=True, density=2)
    plt.show()

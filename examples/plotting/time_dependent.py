import matplotlib.pyplot as plt
import numpy as np

from geodesic_shooting.core import VectorField, TimeDependentVectorField


if __name__ == "__main__":
    n = 20
    nt = 20
    shape = (n, 2*n)
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, shape[0]), np.linspace(-3, 3, shape[1]), indexing='ij')

    def f(x, y, t=0.):
        return np.stack([(0.75*(1 - x) / np.exp(np.sqrt((1 - x)**2 + y**2))
                          + 100*(10 - x) / np.exp(np.sqrt((10 - x)**2 + y**2))) / 6,
                         -y / np.exp(np.sqrt((1 - x)**2 + y**2)) / 6 * t],
                        axis=-1)

    time_series = [VectorField(data=f(grid_x, grid_y, t)) for t in np.linspace(0., 2., nt)]
    time_dependent_vector_field = TimeDependentVectorField(data=time_series)
    magnitude_series = time_dependent_vector_field.get_magnitude_series()
    x_component_series = time_dependent_vector_field.get_component_as_function_series(0)
    y_component_series = time_dependent_vector_field.get_component_as_function_series(1)

    anim1 = time_dependent_vector_field.animate(title="Animation of time-dependent vector field")
    anim2 = magnitude_series.animate(title="Magnitude of vector field")
    anim3 = x_component_series.animate(title="x-component of vector field")
    anim4 = y_component_series.animate(title="y-component of vector field")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

from geodesic_shooting.core import VectorField, TimeDependentVectorField


if __name__ == "__main__":
    n = 20
    shape = (n, 2*n)

    def f(x, y):
        return np.stack([0.8*np.exp(-x**2-y**2) * shape[0] / 6, -0.4*np.exp(-x**2-y**2) * shape[1] / 6], axis=-1)

    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, shape[0]), np.linspace(-3, 3, shape[1]), indexing='ij')

    displacement_field = VectorField(data=f(grid_x, grid_y))
    time_steps = 30
    time_dependent_field = TimeDependentVectorField(data=[displacement_field]*time_steps)
    diffeomorphism = time_dependent_field.integrate()
    inverse_diffeomorphism = time_dependent_field.integrate_backward()
    diffeomorphism.plot_as_warpgrid(title="Diffeomorphism", interval=2)
    inverse_diffeomorphism.plot_as_warpgrid(title="Inverse diffeomorphism")

    approximate_identity = diffeomorphism.push_forward(inverse_diffeomorphism)
    approximate_identity.plot_as_warpgrid(title="Diffeomorphism composed with inverse diffeomorphism")
    approximate_identity_2 = inverse_diffeomorphism.push_forward(diffeomorphism)
    approximate_identity_2.plot_as_warpgrid(title="Inverse diffeomorphism composed with diffeomorphism")

    time_dependent_diffeomorphism = time_dependent_field.integrate(get_time_dependent_diffeomorphism=True)
    time_dependent_diffeomorphism.plot(title="Time dependent diffeomorphism", interval=2, frequency=5)

    time_dependent_i_diffeomorphism = time_dependent_field.integrate_backward(get_time_dependent_diffeomorphism=True)
    time_dependent_i_diffeomorphism.plot(title="Time dependent inverse diffeomorphism", interval=2, frequency=5)

    plt.show()

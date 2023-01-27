import matplotlib.pyplot as plt
import numpy as np

from geodesic_shooting.core import ScalarFunction, VectorField, TimeDependentVectorField


if __name__ == "__main__":
    n = 20
    shape = (n, 2*n)

    def f(x, y):
        return np.stack([0.8*np.exp(-x**2-y**2) * shape[0] / 6, -0.4*np.exp(-x**2-y**2) * shape[1] / 6], axis=-1)

    def g(x, y):
        return np.exp(-x**2-y**2)

    fig = plt.figure()
    axis = fig.subplots(1)
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, shape[0]), np.linspace(-3, 3, shape[1]), indexing='ij')
    function = ScalarFunction(data=g(grid_x, grid_y))
    function.plot(axis=axis, extent=(0, shape[0]-1, 0, shape[1]-1))
    zero_vector_field = VectorField(spatial_shape=shape)
    zero_vector_field.plot_as_warpgrid(title="Original function on identity grid", axis=axis)
    plt.show()

    fig = plt.figure()
    axis = fig.subplots(1)
    displacement_field = VectorField(data=f(grid_x, grid_y))
    time_steps = 30
    time_dependent_field = TimeDependentVectorField(data=[displacement_field]*time_steps)
    diffeomorphism = time_dependent_field.integrate()
    inverse_diffeomorphism = time_dependent_field.integrate_backward()
    function2 = function.push_forward(inverse_diffeomorphism)
    displacement_field.plot_as_warpgrid(axis=axis)
    function2.plot(title="Gaussian deformation", axis=axis, extent=(0, shape[0]-1, 0, shape[1]-1))
    plt.show()

    fig = plt.figure()
    axis = fig.subplots(1)
    function2 = function2.push_forward(diffeomorphism)
    zero_vector_field.plot_as_warpgrid(axis=axis)
    function2.plot(title="Inverted Gaussian deformation", axis=axis, extent=(0, shape[0]-1, 0, shape[1]-1))
    plt.show()

    (function - function2).abs().plot(title="Absolute difference", colorbar=True, extent=(0, shape[0]-1, 0, shape[1]-1))
    plt.show()

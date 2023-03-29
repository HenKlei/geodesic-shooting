import matplotlib.pyplot as plt
import numpy as np

from geodesic_shooting.core import VectorField, ScalarFunction
from geodesic_shooting.utils.helper_functions import tuple_product


if __name__ == "__main__":
    n = 200
    shape = (n, 2*n)

    # ---------------------- Gradient computation ----------------------

    def g(x, y):
        return np.exp(-(6.*x-3.)**2-2.*(6.*y-3.)**2)

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing='ij')

    function = ScalarFunction(data=g(grid_x, grid_y))

    fig, axis = plt.subplots()
    _, vals = function.plot(axis=axis)
    function.grad.plot(title="Finite difference approximation of the gradient vector field of the function",
                       axis=axis, interval=10, scale=None)
    fig.colorbar(vals)
    plt.show()

    def g_gradient(x, y):
        return np.stack([-12.*(6.*x-3.)*g(x, y) / shape[0], -24.*(6.*y-3.)*g(x, y) / shape[1]], axis=-1)

    true_gradient = VectorField(data=g_gradient(grid_x, grid_y))
    fig, axis = plt.subplots()
    _, vals = function.plot(axis=axis)
    true_gradient.plot(title="Exact gradient vector field of the function",
                       axis=axis, interval=10, scale=None)
    fig.colorbar(vals)
    plt.show()

    print(f"Error in the gradient: {(function.grad - true_gradient).norm / tuple_product(shape)}")

    # ---------------------- Divergence computation ----------------------

    def f(x, y):
        return np.stack([np.sin(x + y - 1.), np.cos(1.5 + x - y)], axis=-1)

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing='ij')

    displacement_field = VectorField(data=f(grid_x, grid_y))

    fig, axis = plt.subplots()
    _, vals = displacement_field.div.plot(axis=axis)
    displacement_field.plot("Finite difference approximation of the divergence of the displacement vector field",
                            axis=axis, interval=10, scale=None)
    fig.colorbar(vals)
    plt.show()

    def f_divergence(x, y):
        return np.cos(1 - x - y) / shape[0] + np.sin(1.5 + x - y) / shape[1]

    true_divergence = ScalarFunction(data=f_divergence(grid_x, grid_y))
    fig, axis = plt.subplots()
    _, vals = true_divergence.plot(axis=axis)
    displacement_field.plot("Exact divergence of the displacement vector field", axis=axis, interval=10, scale=None)
    fig.colorbar(vals)
    plt.show()

    fig, axis = plt.subplots()
    _, vals = (true_divergence - displacement_field.div).abs().plot(axis=axis)
    displacement_field.plot("Absolute error of the divergence approximation", axis=axis, interval=10, scale=None)
    fig.colorbar(vals)
    plt.show()

    print(f"Error in the divergence: {(displacement_field.div - true_divergence).norm / tuple_product(shape)}")

import numpy as np

from geodesic_shooting.core import ScalarFunction, VectorField
from geodesic_shooting.utils.grad import finite_difference
from geodesic_shooting.utils.helper_functions import tuple_product


def test_grad():
    shape = (5, 10)

    f1 = ScalarFunction(shape)
    f1[..., 2] = 1
    derivative = VectorField(shape)
    derivative[:, 1, 1] = 0.5
    derivative[:, 3, 1] = -0.5
    assert finite_difference(f1) == derivative

    f2 = ScalarFunction(shape)
    f2[2, ...] = 1
    derivative = VectorField(shape)
    derivative[1, :, 0] = 0.5
    derivative[3, :, 0] = -0.5
    assert finite_difference(f2) == derivative

    v = VectorField(shape)
    v[..., 0] = f1
    v[..., 1] = f2
    assert (finite_difference(v) == np.stack([finite_difference(f1).to_numpy(),
                                              finite_difference(f2).to_numpy()], axis=-1)).all()


def test_differential_operators():
    n = 200
    shape = (n, 2*n)

    def g(x, y):
        return np.exp(-(6.*x-3.)**2-2.*(6.*y-3.)**2)

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing='ij')

    function = ScalarFunction(data=g(grid_x, grid_y))

    def g_gradient(x, y):
        return np.stack([-12.*(6.*x-3.)*g(x, y) / shape[0], -24.*(6.*y-3.)*g(x, y) / shape[1]], axis=-1)

    true_gradient = VectorField(data=g_gradient(grid_x, grid_y))

    assert (function.grad - true_gradient).norm / tuple_product(shape) < 1e-7

    def f(x, y):
        return np.stack([np.sin(x + y - 1.), np.cos(1.5 + x - y)], axis=-1)

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing='ij')

    displacement_field = VectorField(data=f(grid_x, grid_y))

    def f_divergence(x, y):
        return np.cos(1 - x - y) / shape[0] + np.sin(1.5 + x - y) / shape[1]

    true_divergence = ScalarFunction(data=f_divergence(grid_x, grid_y))

    assert (displacement_field.div - true_divergence).norm / tuple_product(shape) < 1e-6

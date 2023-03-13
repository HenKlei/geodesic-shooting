import numpy as np

from geodesic_shooting.core import ScalarFunction, VectorField, TimeDependentVectorField
from geodesic_shooting.utils import grid
from geodesic_shooting.utils.helper_functions import tuple_product


def test_functions():
    shape = (5, 10)
    f = ScalarFunction(shape)

    assert isinstance(f.to_numpy(), np.ndarray)
    assert f.spatial_shape == f.full_shape == shape
    assert f.size == tuple_product(shape) == 50
    assert np.isclose(f.norm, 0.)

    f[0, 0] = 10.
    assert np.isclose(f.norm, 10.)
    assert np.isclose(f.get_norm(order=np.inf), 10.)
    assert np.isclose(f.get_norm(order=5), 10.)

    g = f.copy()
    assert np.isclose((f - g).norm, 0.)
    assert np.isclose((g - f).norm, 0.)
    assert np.isclose((f + g).norm, 20.)
    assert np.isclose((g + f).norm, 20.)
    assert np.isclose((2. * f + g).norm, 30.)


def test_vector_fields():
    shape = (5, 10)
    v = VectorField(shape)

    assert isinstance(v.to_numpy(), np.ndarray)
    assert v.spatial_shape == shape
    assert v.full_shape == (5, 10, 2)
    assert np.isclose(v.norm, 0.)

    v[0, 0, 0] = 10.
    assert np.isclose(v.norm, 10.)
    assert np.isclose(v.get_norm(order=np.inf), 10.)
    assert np.isclose(v.get_norm(order=5), 10.)

    w = v.copy()
    assert np.isclose((v - w).norm, 0.)
    assert np.isclose((w - v).norm, 0.)
    assert np.isclose((v + w).norm, 20.)
    assert np.isclose((w + v).norm, 20.)
    assert np.isclose((2. * v + w).norm, 30.)


def test_vector_fields_plots():
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

    displacement_field.save_vtk("displacement_field", data_label="Displacement vector field")

    displacement_field.plot_as_warpgrid(title="Warpgrid", show_displacement_vectors=True)
    displacement_field.plot(title="Displacement vector field")
    displacement_field.plot(title="Displacement vector field with colors", color_length=True)
    displacement_field.plot_streamlines(title="Streamlines")
    displacement_field.plot_streamlines(title="Streamlines with colors", color_length=True, density=2)
    magnitude.plot(title="Magnitude of vector field")
    x_component.plot(title="x-component of vector field")
    y_component.plot(title="y-component of vector field")


def test_time_integration():
    shape = (10, 20)
    v = VectorField(shape)

    time_steps = 10
    vf_list = [v] * time_steps
    constant_vector_field = TimeDependentVectorField(spatial_shape=shape, time_steps=time_steps, data=vf_list)

    diffeomorphism = constant_vector_field.integrate()
    diffeomorphism_as_vector_field = VectorField(data=diffeomorphism)
    identity_grid = grid.coordinate_grid(shape)
    identity_diffeomorphism = grid.identity_diffeomorphism(shape)
    assert np.isclose((diffeomorphism_as_vector_field - identity_grid).norm, 0.)

    original_translation_vector = np.array([1, 2])
    translation_vector = original_translation_vector / 10
    v += translation_vector
    vf_list = [v] * time_steps
    constant_vector_field = TimeDependentVectorField(spatial_shape=shape, time_steps=time_steps, data=vf_list)
    diffeomorphism = constant_vector_field.integrate()
    diffeomorphism_as_vector_field = identity_grid + (VectorField(data=diffeomorphism) - identity_grid) * 10

    assert np.isclose((diffeomorphism_as_vector_field.push_forward(identity_diffeomorphism)
                       - diffeomorphism_as_vector_field).norm, 0.)
    restriction = np.s_[:-original_translation_vector[0], :-original_translation_vector[1]]
    assert np.isclose((diffeomorphism_as_vector_field - (identity_grid + 10 * v)).get_norm(restriction=restriction), 0.)
    inverse_diffeomorphism = constant_vector_field.integrate_backward()
    restriction = np.s_[original_translation_vector[0]:, original_translation_vector[1]:]

    function = ScalarFunction(spatial_shape=shape)
    function[5, 10] = 1.
    transformed_function = function.push_forward(diffeomorphism)
    assert np.isclose(abs(transformed_function.norm - 1.), 0.)
    inverse_transformed_function = transformed_function.push_forward(inverse_diffeomorphism)
    assert np.isclose((inverse_transformed_function - function).norm, 0.)

    time_steps = 4
    v0 = VectorField(shape)
    t0 = np.array([4, 0])
    v0 += t0
    v1 = VectorField(shape)
    t1 = np.array([4, 8])
    v1 += t1
    v2 = VectorField(shape)
    t2 = np.array([-4, 8])
    v2 += t2
    v3 = VectorField(shape)
    t3 = np.array([4, 0])
    v3 += t3
    vf_list = [v0, v1, v2, v3]
    tdvf = TimeDependentVectorField(spatial_shape=shape, time_steps=time_steps, data=vf_list)
    diffeomorphism = tdvf.integrate()
    inverse_diffeomorphism = tdvf.integrate_backward()

    transformed_function = function.push_forward(diffeomorphism)
    inverse_transformed_function = transformed_function.push_forward(inverse_diffeomorphism)

    data = identity_grid.to_numpy() - np.array([(shape[0]-1)/2, (shape[1]-1)/2])
    data = np.stack([data[..., 1], -data[..., 0]], axis=-1) / 20
    v = VectorField(shape, data=data)
    vf_list = [v] * time_steps
    constant_vector_field = TimeDependentVectorField(spatial_shape=shape, time_steps=time_steps, data=vf_list)
    diffeomorphism = constant_vector_field.integrate()
    inverse_diffeomorphism = constant_vector_field.integrate_backward()

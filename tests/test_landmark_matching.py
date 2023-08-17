import numpy as np
import pytest

import geodesic_shooting
from geodesic_shooting.utils.kernels import GaussianKernel


def test_kernel_gramian():
    positions = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    num_elements, dim = positions.shape
    size = num_elements * dim
    kernel = GaussianKernel()
    mat = []
    pos = positions.reshape((size // dim, dim))

    for i in range(size // dim):
        mat_row = []
        for j in range(size // dim):
            mat_row.append(kernel(pos[i], pos[j]))
        mat.append(mat_row)

    block_mat = np.block(mat)
    assert block_mat.shape == (size, size)

    block_mat_2 = kernel.apply_vectorized(positions, positions, dim)
    assert block_mat_2.shape == (size, size)

    assert np.allclose(block_mat, block_mat_2)


def test_landmark_matching():
    def compute_average_distance(target_landmarks, registered_landmarks):
        dist = 0.
        for x, y in zip(registered_landmarks, target_landmarks):
            dist += np.linalg.norm(x - y)
        dist /= registered_landmarks.shape[0]
        return dist

    input_landmarks = np.array([[5., 3.]])
    target_landmarks = np.array([[6., 2.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.001, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-5

    input_landmarks = np.array([[5., 3.]])
    target_landmarks = np.array([[6., 2.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.001, return_all=True,
                         landmarks_labeled=True, optimization_method='newton')
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-5

    input_landmarks = np.array([[5., 3.], [1., 1.]])
    target_landmarks = np.array([[6., 2.], [0., 0.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.001, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-5

    input_landmarks = np.array([[5., 3.], [1., 1.]])
    target_landmarks = np.array([[6., 2.], [0., 0.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.001, return_all=True,
                         landmarks_labeled=False)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-3

    input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])

    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={'sigma': .5})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.01, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-2

    input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])

    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={'sigma': .5})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.01, return_all=True,
                         landmarks_labeled=True, optimization_method='newton')
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-2


def test_landmark_matching_energies():
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': 1.5},
                                            dim=2, num_landmarks=2)
    kernel = gs.kernel
    input_landmarks = np.array([[0., 0.], [10., 5.]])
    true_momenta = np.array(np.array([[5., 2.5], [-5., -2.5]]))
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(true_momenta, input_landmarks)
    target_landmarks = time_evolution_positions[-1].reshape(input_landmarks.shape)

    # perform the registration using landmark shooting algorithm
    result = gs.register(input_landmarks, target_landmarks,
                         sigma=1e-6, return_all=True, landmarks_labeled=True,
                         initial_momenta=true_momenta)
    final_momenta = result['initial_momenta']

    assert np.allclose(final_momenta, true_momenta)
    assert np.isclose(result['energy_l2'], 0.)

    hamiltonian = 0.
    for pa, qa in zip(true_momenta, input_landmarks):
        for pb, qb in zip(true_momenta, input_landmarks):
            hamiltonian += 0.5 * pa @ kernel(qa, qb) @ pb
    assert np.isclose(result['energy_regularizer'], hamiltonian)

    time_integration_hamiltonian = 0.
    num_timesteps = result['time_evolution_momenta'].shape[0]
    for ps, qs in zip(result['time_evolution_momenta'], result['time_evolution_positions']):
        for pa, qa in zip(ps.reshape(input_landmarks.shape), qs.reshape(input_landmarks.shape)):
            for pb, qb in zip(ps.reshape(input_landmarks.shape), qs.reshape(input_landmarks.shape)):
                time_integration_hamiltonian += 0.5 * pa @ kernel(qa, qb) @ pb / num_timesteps
    assert np.abs(time_integration_hamiltonian - hamiltonian) / hamiltonian < 1e-2

    assert np.average(np.linalg.norm(final_momenta - true_momenta, axis=-1)) < 1e-10


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("num_positions", [1, 2, 5])
@pytest.mark.parametrize("sigma", [2. * (1 - x) for x in np.random.rand(3)])
@pytest.mark.parametrize("sigma_matching", [1., 0.1])
def test_gradient_computations_with_inefficient_version(dim, num_positions, sigma, sigma_matching):
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': sigma},
                                            dim=dim, num_landmarks=num_positions)

    initial_positions = np.random.rand(num_positions, dim)
    targets = np.random.rand(num_positions, dim)
    momenta = np.random.rand(num_positions, dim)

    def compute_matching_function(p, t):
        return np.linalg.norm(p.flatten() - t.flatten()) ** 2

    def compute_gradient_matching_function(p, t):
        return 2. * (p.flatten() - t.flatten())

    momenta_time_dependent, positions_time_dependent = gs.integrate_forward_Hamiltonian(momenta, initial_positions)
    positions = positions_time_dependent[-1]
    d_positions_1, _ = gs.integrate_forward_variational_Hamiltonian(momenta_time_dependent, positions_time_dependent)

    grad_test = np.zeros((gs.num_landmarks, gs.dim))
    assert momenta.shape == (gs.num_landmarks, gs.dim)
    assert initial_positions.shape == (gs.num_landmarks, gs.dim)
    for c, (pc, qc) in enumerate(zip(momenta, initial_positions)):
        for j in range(gs.dim):
            for a, (pa, qa) in enumerate(zip(momenta, initial_positions)):
                for i in range(gs.dim):
                    grad_test[c, j] += gs.kernel(qa, qc)[i, j] * pa[i] * pc[j]

    assert positions.shape == (gs.num_landmarks, gs.dim)
    assert targets.shape == (gs.num_landmarks, gs.dim)
    for c in range(gs.num_landmarks):
        for j in range(gs.dim):
            for a, (qa1, target_qa1) in enumerate(zip(positions, targets)):
                for i in range(gs.dim):
                    grad_test[c, j] += compute_gradient_matching_function(qa1[i], target_qa1[i]) * d_positions_1[a, i, c, j] / sigma_matching ** 2

    _, grad = gs.energy_and_gradient(momenta, initial_positions, targets, sigma_matching, compute_matching_function,
                                     compute_gradient_matching_function, compute_grad=True, return_all_energies=False)

    assert np.allclose(grad, grad_test)

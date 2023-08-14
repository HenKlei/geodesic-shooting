import numpy as np
import pytest

import geodesic_shooting
from geodesic_shooting.utils.kernels import GaussianKernel
from geodesic_shooting.utils.time_integration import ExplicitEuler


def test_Hamiltonian_computation():
    landmarks = np.array([[0., 0.5], [0., 0.65]])
    momenta = np.array([[15., 0.], [15., 0.]])

    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': 1.},
                                            dim=2, num_landmarks=2)
    assert gs.size == 4

    c = np.exp(-0.15 ** 2)
    assert np.isclose(gs.compute_Hamiltonian(momenta, landmarks), (1.+c) * 225)

    def compute_Hamiltonian_explicitly(momenta_in, landmarks_in):
        return 0.5 * ((momenta_in[0, 0]**2 + momenta_in[1, 0]**2 + momenta_in[0, 1]**2 + momenta_in[1, 1]**2)
                      + np.exp(-np.linalg.norm(landmarks_in[0] - landmarks_in[1])**2)
                      * (2. * momenta_in[0, 0] * momenta_in[1, 0] + 2. * momenta_in[1, 1] * momenta_in[0, 1]))

    assert np.isclose(gs.compute_Hamiltonian(momenta, landmarks), compute_Hamiltonian_explicitly(momenta, landmarks))

    landmarks = np.random.rand(2, 2)
    momenta = np.random.rand(2, 2)
    assert np.isclose(gs.compute_Hamiltonian(momenta, landmarks), compute_Hamiltonian_explicitly(momenta, landmarks))


@pytest.mark.parametrize("sigma", [2. * (1 - x) for x in np.random.rand(10)])
def test_rhs_position_function(sigma):
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': sigma},
                                            dim=2, num_landmarks=2)

    landmarks = np.random.rand(2, 2)
    momenta = np.random.rand(2, 2)

    rhs = gs._rhs_position_function(landmarks.flatten(), momenta.flatten())
    assert rhs.shape == (gs.size,) == (4,)
    rhs_explicit = np.zeros(rhs.shape)
    rhs_explicit[:2] = (gs.kernel(landmarks[0], landmarks[0]) @ momenta[0]
                        + gs.kernel(landmarks[0], landmarks[1]) @ momenta[1])
    rhs_explicit[2:] = (gs.kernel(landmarks[1], landmarks[0]) @ momenta[0]
                        + gs.kernel(landmarks[1], landmarks[1]) @ momenta[1])
    assert np.allclose(rhs, rhs_explicit)

    rhs_more_explicit = np.zeros(rhs.shape)
    rhs_more_explicit[:2] = momenta[0] + np.exp(-sigma * np.linalg.norm(landmarks[0] - landmarks[1])**2) * momenta[1]
    rhs_more_explicit[2:] = momenta[1] + np.exp(-sigma * np.linalg.norm(landmarks[0] - landmarks[1])**2) * momenta[0]
    assert np.allclose(rhs, rhs_more_explicit)


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("num_landmarks", [1, 2, 5, 10, 20])
@pytest.mark.parametrize("sigma", [2. * (1 - x) for x in np.random.rand(3)])
def test_rhs_position_function_with_inefficient_version(dim, num_landmarks, sigma):
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': sigma},
                                            dim=dim, num_landmarks=num_landmarks)

    def working_inefficient_rhs_position_function(positions, momenta):
        rhs = np.zeros((gs.num_landmarks, gs.dim))
        for a, qa in enumerate(positions.reshape((gs.num_landmarks, gs.dim))):
            for b, (pb, qb) in enumerate(zip(momenta.reshape((gs.num_landmarks, gs.dim)),
                                             positions.reshape((gs.num_landmarks, gs.dim)))):
                rhs[a] += gs.kernel(qa, qb) @ pb
        return rhs.flatten()

    landmarks = np.random.rand(num_landmarks, dim)
    momenta = np.random.rand(num_landmarks, dim)
    rhs = gs._rhs_position_function(landmarks.flatten(), momenta.flatten())
    rhs_working_inefficient = working_inefficient_rhs_position_function(landmarks, momenta)
    assert np.allclose(rhs, rhs_working_inefficient)


def test_rhs_momenta_function():
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': 1.}, dim=2, num_landmarks=2)

    landmarks = np.random.rand(2, 2)
    momenta = np.random.rand(2, 2)

    rhs = gs._rhs_momenta_function(momenta.flatten(), landmarks.flatten())
    assert rhs.shape == (gs.size,) == (4,)
    rhs_explicit = np.zeros(rhs.shape)
    rhs_explicit[0] = - (- 2. * (landmarks[0, 0] - landmarks[0, 0]) * gs.kernel(landmarks[0], landmarks[0])[0, 0]
                         * momenta[0, 0]**2
                         - 2. * (landmarks[0, 0] - landmarks[1, 0]) * gs.kernel(landmarks[0], landmarks[1])[0, 0]
                         * momenta[0, 0] * momenta[1, 0]
                         - 2. * (landmarks[0, 0] - landmarks[0, 0]) * gs.kernel(landmarks[0], landmarks[0])[1, 1]
                         * momenta[0, 1]**2
                         - 2. * (landmarks[0, 0] - landmarks[1, 0]) * gs.kernel(landmarks[0], landmarks[1])[1, 1]
                         * momenta[0, 1] * momenta[1, 1])
    rhs_explicit[1] = - (- 2. * (landmarks[0, 1] - landmarks[0, 1]) * gs.kernel(landmarks[0], landmarks[0])[0, 0]
                         * momenta[0, 0]**2
                         - 2. * (landmarks[0, 1] - landmarks[1, 1]) * gs.kernel(landmarks[0], landmarks[1])[0, 0]
                         * momenta[0, 0] * momenta[1, 0]
                         - 2. * (landmarks[0, 1] - landmarks[0, 1]) * gs.kernel(landmarks[0], landmarks[0])[1, 1]
                         * momenta[0, 1]**2
                         - 2. * (landmarks[0, 1] - landmarks[1, 1]) * gs.kernel(landmarks[0], landmarks[1])[1, 1]
                         * momenta[0, 1] * momenta[1, 1])
    rhs_explicit[2] = - (- 2. * (landmarks[1, 0] - landmarks[1, 0]) * gs.kernel(landmarks[1], landmarks[1])[0, 0]
                         * momenta[1, 0]**2
                         - 2. * (landmarks[1, 0] - landmarks[0, 0]) * gs.kernel(landmarks[1], landmarks[0])[0, 0]
                         * momenta[1, 0] * momenta[0, 0]
                         - 2. * (landmarks[1, 0] - landmarks[1, 0]) * gs.kernel(landmarks[1], landmarks[1])[1, 1]
                         * momenta[1, 1]**2
                         - 2. * (landmarks[1, 0] - landmarks[0, 0]) * gs.kernel(landmarks[1], landmarks[0])[1, 1]
                         * momenta[1, 1] * momenta[0, 1])
    rhs_explicit[3] = - (- 2. * (landmarks[1, 1] - landmarks[1, 1]) * gs.kernel(landmarks[1], landmarks[1])[0, 0]
                         * momenta[1, 0]**2
                         - 2. * (landmarks[1, 1] - landmarks[0, 1]) * gs.kernel(landmarks[1], landmarks[0])[0, 0]
                         * momenta[1, 0] * momenta[0, 0]
                         - 2. * (landmarks[1, 1] - landmarks[1, 1]) * gs.kernel(landmarks[1], landmarks[1])[1, 1]
                         * momenta[1, 1]**2
                         - 2. * (landmarks[1, 1] - landmarks[0, 1]) * gs.kernel(landmarks[1], landmarks[0])[1, 1]
                         * momenta[1, 1] * momenta[0, 1])
    assert np.allclose(rhs, rhs_explicit)

    rhs_more_complex = np.zeros(gs.size)
    for a, (pa, qa) in enumerate(zip(momenta, landmarks)):
        for i in range(gs.dim):
            for b, (pb, qb) in enumerate(zip(momenta, landmarks)):
                for j in range(gs.dim):
                    for k in range(gs.dim):
                        rhs_more_complex[a*gs.dim + i] -= gs.kernel.derivative_1(qa, qb, i)[j, k] * pa[j] * pb[k]

    assert np.allclose(rhs, rhs_more_complex)


def test_integrate_forward_Hamiltonian_single_step_explicit_Euler():
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': 1. / np.sqrt(2.)},
                                            dim=2, num_landmarks=2, time_integrator=ExplicitEuler, time_steps=2)

    landmarks = np.random.rand(2, 2)
    momenta = np.random.rand(2, 2)

    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)
    assert len(time_evolution_positions) == len(time_evolution_momenta) == 2
    assert np.allclose(time_evolution_momenta[0], momenta.flatten())
    assert np.allclose(time_evolution_positions[0], landmarks.flatten())

    rhs = gs._rhs_momenta_function(momenta.flatten(), landmarks.flatten())
    assert np.allclose(time_evolution_momenta[-1], momenta.flatten() + rhs)

    rhs = gs._rhs_position_function(landmarks.flatten(), momenta.flatten())
    assert np.allclose(time_evolution_positions[-1], landmarks.flatten() + rhs)


def test_integrate_forward_Hamiltonian_single_landmark():
    gs = geodesic_shooting.LandmarkShooting(dim=2, num_landmarks=1)

    landmarks = np.array([[1., 1.]])
    momenta = np.array([[2., 3.]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)

    assert np.allclose(time_evolution_momenta[-1], momenta)
    assert np.allclose(time_evolution_positions[-1], landmarks + momenta)

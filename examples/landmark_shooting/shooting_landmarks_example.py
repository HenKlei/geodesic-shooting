import matplotlib.pyplot as plt
import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.kernels import GaussianKernel
from geodesic_shooting.utils.visualization import plot_landmark_trajectories


if __name__ == "__main__":
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': 1.5},
                                            dim=2, num_landmarks=2)

    landmarks = np.array([[0., 0.], [0., 0.15]])
    momenta = np.array([[15., 0.], [15., 0.]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=gs.kernel,
                               show_vector_fields=False)
    plt.show()

    landmarks = np.array([[-0.4, -0.125], [0.4, 0.125]])
    momenta = np.array([[2., 0.], [-2., 0.]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=gs.kernel,
                               show_vector_fields=False)
    plt.show()

    landmarks = np.array([[0., 0.], [10., 10.]])
    momenta = np.array([[5., 5.], [-5., -5.]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=gs.kernel,
                               show_vector_fields=False)
    plt.show()

    landmarks = np.array([[0., 0.], [10., 5.]])
    momenta = np.array([[5., 2.5], [-5., -2.5]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=gs.kernel,
                               show_vector_fields=False)
    plt.show()

    landmarks = np.array([[0., 0.], [10., 10.]])
    momenta = np.array([[10., 0.], [0., -10.]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=gs.kernel,
                               show_vector_fields=False)
    plt.show()

    landmarks = np.array([[0., 0.], [10., 10.]])
    momenta = np.array([[20., 0.], [0., -20.]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=gs.kernel,
                               show_vector_fields=False)
    plt.show()

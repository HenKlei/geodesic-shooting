import numpy as np
from matplotlib import pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.kernels import GaussianKernel
from geodesic_shooting.utils.visualization import plot_landmark_trajectories


if __name__ == "__main__":
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': 16},
                                            dim=2, num_landmarks=2)

    landmarks = np.array([[0.1, 0.5-0.125], [0.9, 0.5+0.125]])
    force = 0.6
    momenta = np.array([[force, 0.], [-force, 0.]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(momenta, landmarks)
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=gs.kernel,
                               show_vector_fields=False)

    vf = gs.get_vector_field(momenta, landmarks)
    vf.plot(color_length=True)

    time_dependent_flow = gs.compute_diffeomorphism(momenta, landmarks, get_time_dependent_diffeomorphism=True)
    time_dependent_flow[-1].plot()
    _ = time_dependent_flow.animate()
    plt.show()

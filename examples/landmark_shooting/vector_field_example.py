import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting

from geodesic_shooting.utils.visualization import (animate_landmark_trajectories,
                                                   plot_initial_momenta_and_landmarks,
                                                   plot_landmark_trajectories)


if __name__ == "__main__":
    input_landmarks = np.array([[-1., 0.], [-0.5, 1.], [1., 0.]])

    gs = geodesic_shooting.LandmarkShooting(dim=2, num_landmarks=input_landmarks.shape[0])

    initial_momenta = np.array([[1./np.sqrt(2.), 1./np.sqrt(2.)],
                                [-2./np.sqrt(5.), 1./np.sqrt(5.)],
                                [-2./np.sqrt(5.), -1./np.sqrt(5.)]])

    N = 30
    min_x = -2.
    max_x = 2.
    min_y = -2.
    max_y = 2.

    plot_initial_momenta_and_landmarks(initial_momenta.flatten(), input_landmarks.flatten(), kernel=gs.kernel,
                                       min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, N=N)
    plt.show()

    momenta, positions = gs.integrate_forward_Hamiltonian(initial_momenta.flatten(),
                                                          input_landmarks.flatten())

    plot_landmark_trajectories(momenta, positions, kernel=gs.kernel,
                               min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, N=N)
    plt.show()

    ani = animate_landmark_trajectories(momenta, positions, kernel=gs.kernel,
                                        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, N=N)
    plt.show()

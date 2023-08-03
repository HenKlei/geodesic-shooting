import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting

from geodesic_shooting.utils.visualization import (animate_landmark_trajectories,
                                                   plot_initial_momenta_and_landmarks,
                                                   plot_landmark_trajectories)


if __name__ == "__main__":
    input_landmarks = np.array([[0.25, 0.5], [3./8., 0.75], [0.75, 0.5]])

    gs = geodesic_shooting.LandmarkShooting(dim=2, num_landmarks=input_landmarks.shape[0],
                                            kwargs_kernel={'sigma': 1.})

    initial_momenta = np.array([[(2.+1./np.sqrt(2.))/4., (2.+1./np.sqrt(2.))/4.],
                                [(2.-2./np.sqrt(5.))/4., (2.+1./np.sqrt(5.))/4.],
                                [(2.-2./np.sqrt(5.))/4., (2.-1./np.sqrt(5.))/4.]]) / 5.

    N = 30

    plot_initial_momenta_and_landmarks(initial_momenta, input_landmarks, kernel=gs.kernel, N=N)
    plt.show()

    momenta, positions = gs.integrate_forward_Hamiltonian(initial_momenta, input_landmarks)

    plot_landmark_trajectories(momenta, positions, kernel=gs.kernel, N=N)
    plt.show()

    ani = animate_landmark_trajectories(momenta, positions, kernel=gs.kernel, N=N)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting

from geodesic_shooting.utils.visualization import (animate_landmark_trajectories,
                                                   plot_initial_momenta_and_landmarks,
                                                   plot_landmark_matchings,
                                                   plot_landmark_trajectories)


if __name__ == "__main__":
    # define greyscale images
    input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    print(input_landmarks.shape)
    print(input_landmarks)
    target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])
    print(target_landmarks.shape)
    print(target_landmarks)

    # perform the registration
    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, return_all=True)
    final_momenta = result['initial_momenta']
    registered_landmarks = result['registered_landmarks']

    plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks)
    plt.show()

    plot_initial_momenta_and_landmarks(final_momenta.flatten(), registered_landmarks.flatten(),
                                       min_x=0., max_x=7., min_y=-2., max_y=4.)
    plt.show()

    time_evolution_momenta = result['time_evolution_momenta']
    time_evolution_positions = result['time_evolution_positions']
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
                               min_x=0., max_x=7., min_y=-2., max_y=4.)
    plt.show()

    ani = animate_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
                                        min_x=0., max_x=7., min_y=-2., max_y=4.)
    plt.show()

    print(f"Input: {input_landmarks}")
    print(f"Target: {target_landmarks}")
    print(f"Result: {registered_landmarks}")
    rel_error = (np.linalg.norm(target_landmarks - registered_landmarks)
                 / np.linalg.norm(target_landmarks))
    print(f"Relative norm of difference: {rel_error}")

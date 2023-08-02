import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting

from geodesic_shooting.utils.visualization import (animate_landmark_trajectories,
                                                   plot_initial_momenta_and_landmarks,
                                                   plot_landmarks, plot_landmark_trajectories)
from geodesic_shooting.core import ScalarFunction


if __name__ == "__main__":
    # define landmark positions
    input_landmarks = np.array([[4./7., 1.75/7.], [3./7., 11./21.], [4./7., 13./21.], [5./7., 5./7.]])
    target_landmarks = np.array([[3.4/7., 1.75/7.], [4./7., 2.5/7.], [5./7., 3.25/7.], [6./7., 4./7.]])

    from geodesic_shooting.utils.kernels import GaussianKernel
    kernel_dist = GaussianKernel(sigma=1., scalar=True)

    def compute_matching_function(positions):
        reshaped_positions = positions.reshape(input_landmarks.shape)
        dist = 0.
        for p in reshaped_positions:
            for q in reshaped_positions:
                dist += kernel_dist(p, q)
            for t in target_landmarks:
                dist -= 2. * kernel_dist(p, t)
        for t in target_landmarks:
            for s in target_landmarks:
                dist += kernel_dist(t, s)
        return dist

    print(f"Matching function value for target landmarks: {compute_matching_function(target_landmarks)}")

    # perform the registration using landmark shooting algorithm
    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={'sigma': 0.1})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.1, return_all=True,
                         landmarks_labeled=False, kwargs_kernel_dist={'sigma': 1.})
    final_momenta = result['initial_momenta']
    registered_landmarks = result['registered_landmarks']

    kernel = gs.kernel

    # plot results
    plot_landmarks(input_landmarks, target_landmarks, registered_landmarks)

    plot_initial_momenta_and_landmarks(final_momenta, registered_landmarks, kernel=kernel)

    time_evolution_momenta = result['time_evolution_momenta']
    time_evolution_positions = result['time_evolution_positions']
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=kernel)

    ani = animate_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=kernel)

    nx = 70
    ny = 60
    spatial_shape = (nx, ny)
    flow = gs.compute_time_evolution_of_diffeomorphisms(final_momenta, input_landmarks,
                                                        spatial_shape=spatial_shape)
    flow.plot("Flow")

    def set_landmarks_in_image(img, landmarks, sigma=1.):
        x, y = np.meshgrid(np.linspace(0., 1., nx), np.linspace(0., 1., ny))
        for i, l in enumerate(landmarks):
            dst = (x - l[0])**2 + (y - l[1])**2
            img += (i + 1.) * np.exp(-(dst.T / (sigma**2))) / (sigma**2)

    sigma = kernel.sigma
    image = ScalarFunction(spatial_shape)
    target_image = ScalarFunction(spatial_shape)
    set_landmarks_in_image(image, input_landmarks, sigma=sigma)
    set_landmarks_in_image(target_image, target_landmarks, sigma=sigma)

    image.plot("Original image")
    target_image.plot("Target image")
    resulting_image = image.push_forward(flow)
    resulting_image.plot("Transformed image")
    plt.show()

    print(f"Input: {input_landmarks}")
    print(f"Target: {target_landmarks}")
    print(f"Result: {registered_landmarks}")
    print(f"Matching function value for registered landmarks: {compute_matching_function(registered_landmarks)}")

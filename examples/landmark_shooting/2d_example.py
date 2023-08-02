import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting

from geodesic_shooting.utils.visualization import (animate_landmark_trajectories,
                                                   plot_initial_momenta_and_landmarks,
                                                   plot_landmark_matchings,
                                                   plot_landmark_trajectories, animate_warpgrids)
from geodesic_shooting.core import ScalarFunction


if __name__ == "__main__":
    # define landmark positions
    input_landmarks = np.array([[5./7., 5./7.], [4./7., 4./7.], [1./7., 2./7.], [2./7., 5./7.]])
    target_landmarks = np.array([[6./7., 4./7.], [5./7., 3./7.], [1./7., 1./7.], [2.5/7., 4./7.]])

    # perform the registration using landmark shooting algorithm
    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={'sigma': 0.1})
    result = gs.register(input_landmarks, target_landmarks, optimization_method='newton',
                         sigma=0.1, return_all=True, landmarks_labeled=True)
    final_momenta = result['initial_momenta']
    registered_landmarks = result['registered_landmarks']

    kernel = gs.kernel

    vf = gs.get_vector_field(final_momenta, result["input_landmarks"])
    vf.plot("Vector field at initial time", color_length=True)
    vf.get_magnitude().plot("Magnitude of vector field at initial time")

    # plot results
    plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks)

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
    error = np.linalg.norm(target_landmarks - registered_landmarks)
    print(f"Norm of difference: {error}")

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
    input_landmarks = np.array([[.1, .1], [.3, .3], [.5, .5], [.7, .7]])
    target_landmarks = np.array([[.2, .1], [.4, .3], [.6, .5], [.8, .7]])

    #input_landmarks = np.array([[.25, .25]])
    #target_landmarks = np.array([[.75, .5]])

    # perform the registration using landmark shooting algorithm
    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={'sigma': 1})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.1, return_all=True, landmarks_labeled=True)
    final_momenta = result['initial_momenta']
    registered_landmarks = result['registered_landmarks']

    # plot results
    plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks)

    plot_initial_momenta_and_landmarks(final_momenta, registered_landmarks,
                                       min_x=0., max_x=1., min_y=0., max_y=1.)

    time_evolution_momenta = result['time_evolution_momenta']
    time_evolution_positions = result['time_evolution_positions']
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
                               min_x=0., max_x=1., min_y=0., max_y=1.)

#    ani = animate_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
#                                        min_x=0., max_x=1., min_y=0., max_y=1.)

    nx = 70
    ny = 60
    mins = np.array([0., 0.])
    maxs = np.array([1., 1.])
    spatial_shape = (nx, ny)
    flow = gs.compute_time_evolution_of_diffeomorphisms(final_momenta, input_landmarks,
                                                        mins=mins, maxs=maxs, spatial_shape=spatial_shape)
    flow.plot("Flow")

    const = 200.

    def set_landmarks_in_image(img, landmarks, sigma=1.):
        x, y = np.meshgrid(np.linspace(mins[0], maxs[0], nx), np.linspace(mins[1], maxs[1], ny))
        for i, l in enumerate(landmarks):
            dst = (x - l[0])**2 + (y - l[1])**2
            img += (i + 1.) * np.exp(-(const * dst.T))

    sigma = gs.kernel.sigma
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

    print(f"Error in image matching: {(target_image - resulting_image).norm / target_image.norm}")

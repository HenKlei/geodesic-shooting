import numpy as np

import geodesic_shooting


def test_1d():
    # define greyscale images
    N = 100
    input_ = np.zeros(N)
    target = np.zeros(N)
    input_[N//5:2*N//5] = 1
    target[2*N//5:3*N//5] = 1

    # perform the registration
    lddmm = geodesic_shooting.LDDMM(alpha=6., exponent=1)
    result = lddmm.register(input_, target, sigma=0.01, epsilon=0.0001, early_stopping=20,
                            return_all=True)

    transformed_input = result['transformed_input']

    assert np.linalg.norm(target - transformed_input) / np.linalg.norm(target) < 1e-3

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=6., exponent=1)
    result = gs.register(input_, target, sigma=0.05, early_stopping=20,
                         parameters_line_search={'min_stepsize': 1e-3, 'max_stepsize': 1e-2},
                         return_all=True)

    transformed_input = result['transformed_input']

    assert np.linalg.norm(target - transformed_input) / np.linalg.norm(target) < 1e-3

    input_landmarks = np.array([[1.], [2.], [9.]])
    target_landmarks = np.array([[3.], [4.5], [8.]])

    # perform the registration using landmark shooting algorithm
    landmark_gs = geodesic_shooting.LandmarkShooting()
    result = landmark_gs.register(input_landmarks, target_landmarks, sigma=0.05, epsilon=0.001, return_all=True)
    registered_landmarks = result['registered_landmarks']

    assert np.linalg.norm(target_landmarks - registered_landmarks) / np.linalg.norm(target_landmarks) < 1e-3

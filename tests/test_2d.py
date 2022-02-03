import numpy as np

import geodesic_shooting


def test_2d():
    # define greyscale images
    N = 10
    M = 5
    input_ = np.zeros((N, M))
    target = np.zeros((N, M))
    input_[N//5:2*N//5, M//5:2*M//5] = 1
    target[2*N//5:3*N//5, M//5:2*M//5] = 1

    # perform the registration with the LDDMM algorithm
    lddmm = geodesic_shooting.LDDMM(alpha=1000., exponent=3)
    result = lddmm.register(input_, target, sigma=0.01,
                            parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1e-4},
                            return_all=True)

    transformed_input = result['transformed_input']

    assert np.abs(np.linalg.norm(target - transformed_input) / np.linalg.norm(target)) < 5e-1

    # perform the registration with the geodesic shooting algorithm
    gs = geodesic_shooting.GeodesicShooting(alpha=1000., exponent=3)
    result = gs.register(input_, target, sigma=0.01,
                         parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 5e-4},
                         return_all=True)

    transformed_input = result['transformed_input']

    assert np.abs(np.linalg.norm(target - transformed_input) / np.linalg.norm(target)) < 5e-1

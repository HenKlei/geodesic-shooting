import numpy as np

import geodesic_shooting


def test_2d():
    # define greyscale images
    N = 20
    M = 10
    input_ = np.zeros((N, M))
    target = np.zeros((N, M))
    input_[N//5:2*N//5, M//5:2*M//5] = 1
    target[2*N//5:3*N//5, M//5:2*M//5] = 1

    # perform the registration with the LDDMM algorithm
    lddmm = geodesic_shooting.LDDMM(alpha=1000., exponent=3)
    result = lddmm.register(input_, target, sigma=0.01, epsilon=0.0001,
                            iterations=100, return_all=True)

    transformed_input = result['transformed_input']

    assert np.abs(np.linalg.norm(target - transformed_input) / np.linalg.norm(target)) < 5e-2

    # perform the registration with the geodesic shooting algorithm
    gs = geodesic_shooting.GeodesicShooting(alpha=1000., exponent=3)
    result = gs.register(input_, target, sigma=0.01, epsilon=0.0001,
                         iterations=200, return_all=True)

    transformed_input = result['transformed_input']

    assert np.abs(np.linalg.norm(target - transformed_input) / np.linalg.norm(target)) < 5e-2

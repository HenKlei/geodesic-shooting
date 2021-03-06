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
    gs = geodesic_shooting.GeodesicShooting(alpha=6., gamma=1.)
    image, v0, energies, Phi0, length = gs.register(input_, target, sigma=0.05, epsilon=0.01, return_all=True)

    assert np.abs(np.linalg.norm(target - image) / np.linalg.norm(target)) < 1e-3

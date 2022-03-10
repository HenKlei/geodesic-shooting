import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle, make_square


def test_translation():
    # create images
    input_ = (make_circle((64, 64), np.array([25, 40]), 18) * 0.2
              + make_circle((64, 64), np.array([25, 40]), 15) * 0.8)
    target = (make_circle((64, 64), np.array([40, 25]), 18) * 0.2
              + make_circle((64, 64), np.array([40, 25]), 15) * 0.8)

    """
    # perform the registration with the LDDMM algorithm
    lddmm = geodesic_shooting.LDDMM(alpha=1000., exponent=1.)
    result = lddmm.register(input_, target, sigma=0.1,
                            parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1e-3},
                            return_all=True)

    assert np.linalg.norm(target - result['transformed_input']) / np.linalg.norm(target) < 1e-1
    """

    # perform the registration with the geodesic shooting algorithm
    gs = geodesic_shooting.GeodesicShooting(alpha=1000., exponent=1.)
    result = gs.register(input_, target, sigma=0.1,
                         parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1.},
                         iterations=100, return_all=True)

    assert (target - result['transformed_input']).norm / target.norm < 5e-2

    # create images
    input_ = make_square((64, 64), np.array([24, 24]), 40)
    target = make_square((64, 64), np.array([40, 32]), 40)

    """
    # perform the registration with the LDDMM algorithm
    lddmm = geodesic_shooting.LDDMM(alpha=1000., exponent=1.)
    result = lddmm.register(input_, target, sigma=0.1,
                            parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1e-3},
                            return_all=True)

    assert np.linalg.norm(target - result['transformed_input']) / np.linalg.norm(target) < 5e-2
    """

    # perform the registration with the geodesic shooting algorithm
    gs = geodesic_shooting.GeodesicShooting(alpha=1000., exponent=1.)
    result = gs.register(input_, target, sigma=0.1,
                         parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1e-1},
                         iterations=50, return_all=True)

    assert (target - result['transformed_input']).norm / target.norm < 5e-2

import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle


def test_circle_scaling():
    # create images
    input_ = make_circle((64, 64), np.array([32, 32]), 10)
    target = make_circle((64, 64), np.array([32, 32]), 20)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=6., exponent=1.)
    result = gs.register(input_, target, sigma=0.1,
                         parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1e-1},
                         iterations=50, return_all=True)

    assert (target - result['transformed_input']).norm / target.norm < 1e-8

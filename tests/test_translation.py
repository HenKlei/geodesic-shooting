import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle, make_square


def test_translation():
    # create images
    template = (make_circle((64, 64), np.array([25, 35]), 15) * 0.2
                + make_circle((64, 64), np.array([25, 35]), 12) * 0.8)
    target = (make_circle((64, 64), np.array([35, 25]), 15) * 0.2
              + make_circle((64, 64), np.array([35, 25]), 12) * 0.8)

    # perform the registration with the geodesic shooting algorithm
    gs = geodesic_shooting.GeodesicShooting(alpha=0.01, exponent=2)
    result = gs.register(template, target, sigma=0.01, return_all=True)

    assert (target - result['transformed_input']).norm / target.norm < 5e-2

    # create images
    template = make_square((64, 64), np.array([24, 24]), 40)
    target = make_square((64, 64), np.array([40, 32]), 40)

    # perform the registration with the geodesic shooting algorithm
    gs = geodesic_shooting.GeodesicShooting(alpha=1., exponent=2)
    result = gs.register(template, target, sigma=0.1, return_all=True)

    assert (target - result['transformed_input']).norm / target.norm < 5e-2

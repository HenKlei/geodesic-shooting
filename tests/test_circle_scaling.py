import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle


def test_circle_scaling():
    # create images
    input_ = make_circle((64, 64), np.array([32, 32]), 10)
    target = make_circle((64, 64), np.array([32, 32]), 20)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=6., exponent=2.)
    result = gs.register(input_, target, sigma=0.1, return_all=True)

    assert (target - result['transformed_input']).norm / target.norm < 1e-3

import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle, make_square


def test_square_to_circle():
    # create images
    target = make_circle((64, 64), np.array([32, 32]), 20)
    template = make_square((64, 64), np.array([32, 32]), 40)

    # set restriction of where to compute the error and the gradient
    restriction = np.s_[2:-20, 2:-20]

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.01, exponent=1)
    result = gs.register(template, target, sigma=0.01, return_all=True, restriction=restriction)

    norm_difference = (target - result['transformed_input']).get_norm(restriction=restriction)
    assert norm_difference / target.get_norm(restriction=restriction) < 1e-1

import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle, make_square


def test_circle_to_square():
    # create images
    template = make_circle((64, 64), np.array([32, 32]), 20)
    target = make_square((64, 64), np.array([32, 32]), 40)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.1, exponent=1, gamma=5.)
    result = gs.register(template, target, sigma=0.01, return_all=True,
                         optimizer_options={'maxiter': 50, 'grad_norm_tol': 1e-3})

    assert (target - result['transformed_input']).norm / target.norm < 1e-2

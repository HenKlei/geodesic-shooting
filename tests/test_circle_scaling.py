import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle


def test_circle_scaling():
    # create images
    template = make_circle((64, 64), np.array([32, 32]), 10)
    target = make_circle((64, 64), np.array([32, 32]), 20)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.01, exponent=1)
    result = gs.register(template, target, sigma=0.01, return_all=True, optimizer_options={'disp': True, 'maxiter': 20})

    assert (target - result['transformed_input']).norm / target.norm < 1e-2

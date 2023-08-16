import numpy as np

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction


def test_3d_geodesic_shooting():
    shape = (10, 10, 10)
    template = np.zeros(shape)
    target = np.zeros(shape)
    template[shape[0]//5:2*shape[0]//5, shape[1]//5:2*shape[1]//5, shape[2]//5:2*shape[2]//5] = 1
    target[shape[0]//5:3*shape[0]//5, shape[1]//5:2*shape[1]//5, shape[2]//5:3*shape[2]//5] = 1
    template = ScalarFunction(data=template)
    target = ScalarFunction(data=target)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.1, exponent=1)
    result = gs.register(template, target, sigma=0.01, return_all=True)

    assert (target - result['transformed_input']).norm / target.norm < 1e-2

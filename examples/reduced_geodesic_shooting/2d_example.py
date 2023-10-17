import numpy as np

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils.summary import plot_registration_results


if __name__ == "__main__":
    # define greyscale images
    N = 10
    M = 5
    template = ScalarFunction((N, M))
    target = ScalarFunction((N, M))
    template[N//5:2*N//5, M//5:2*M//5] = 1
    target[2*N//5:3*N//5, M//5:2*M//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.01, exponent=1)
    result = gs.register(template, target, sigma=0.1, return_all=True)

    #plot_registration_results(result)

    v0 = result['initial_vector_field']
    assert v0 == result['vector_fields'][0]
    v1 = result['vector_fields'][-1]
    rb = [v0, v1 - v0.dot(v1) * v0 / v0.norm**2]
    rb[0] = rb[0] / rb[0].norm
    rb[1] = rb[1] / rb[1].norm
    for vf in rb:
        assert np.isclose(vf.norm, 1.)
    assert np.isclose(rb[0].dot(rb[1]), 0.)
    #reduced_gs = geodesic_shooting.ReducedGeodesicShooting(rb, alpha=0.1, exponent=1)

    #result_reduced = reduced_gs.register(template, target, sigma=0.01, return_all=True)

    #plot_registration_results(result_reduced)

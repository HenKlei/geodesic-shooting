import numpy as np

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils.summary import plot_registration_results


if __name__ == "__main__":
    # define greyscale images
    N = 10
    M = 5
    input_ = ScalarFunction((N, M))
    target = ScalarFunction((N, M))
    input_[N//5:2*N//5, M//5:2*M//5] = 1
    target[2*N//5:3*N//5, M//5:2*M//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=6., exponent=3)
    result = gs.register(input_, target, sigma=0.01, return_all=True)

    plot_registration_results(result)

    v0 = result['initial_vector_field']
    rb = [v0 / v0.norm, ]
    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(rb, alpha=6., exponent=3)

    result_reduced = reduced_gs.register(input_, target, sigma=0.1, return_all=True)

    plot_registration_results(result_reduced)

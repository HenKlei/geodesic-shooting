import numpy as np

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils.summary import plot_registration_results


if __name__ == "__main__":
    shape = (10, 10, 10)
    input_ = np.zeros(shape)
    target = np.zeros(shape)
    input_[shape[0]//5:2*shape[0]//5, shape[1]//5:2*shape[1]//5, shape[2]//5:2*shape[2]//5] = 1
    target[shape[0]//5:3*shape[0]//5, shape[1]//5:2*shape[1]//5, shape[2]//5:3*shape[2]//5] = 1
    input_ = ScalarFunction(data=input_)
    target = ScalarFunction(data=target)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=5., exponent=2)
    result = gs.register(input_, target, sigma=0.01, return_all=True)

    plot_registration_results(result)

import numpy as np

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction


if __name__ == "__main__":
    shape = (10, 10, 10)
    input_ = np.zeros(shape)
    target = np.zeros(shape)
    input_[shape[0]//5:2*shape[0]//5, shape[1]//5:2*shape[1]//5, shape[2]//5:2*shape[2]//5] = 1
    target[shape[0]//5:3*shape[0]//5, shape[1]//5:2*shape[1]//5, shape[2]//5:3*shape[2]//5] = 1
    input_ = ScalarFunction(data=input_)
    target = ScalarFunction(data=target)

    print(f"Initial relative norm of difference: {(target - input_).norm / target.norm}")

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=500., exponent=1)
    result = gs.register(input_, target, sigma=0.05, return_all=True)

    transformed_input = result['transformed_input']

    print(f"Registration finished after {result['iterations']} iterations.")
    print(f"Registration took {result['time']} seconds.")
    print(f"Reason for the registration algorithm to stop: {result['reason_registration_ended']}.")
    print(f"Relative norm of difference: {(target - transformed_input).norm / target.norm}")

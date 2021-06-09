import numpy as np

import geodesic_shooting


if __name__ == "__main__":
    # define greyscale images
    N = 100
    input_ = np.zeros(N)
    target = np.zeros(N)
    input_[N//5:2*N//5] = 1
    target[2*N//5:3*N//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=6., exponent=1.)
    result = gs.register(input_, target, sigma=0.05, epsilon=0.01, early_stopping=20,
                         return_all=True)

    transformed_input = result['transformed_input']

    print(f"Input: {input_}")
    print(f"Target: {target}")
    print(f"Registration result: {transformed_input}")
    print(f"Registration finished after {result['iterations']} iterations.")
    print(f"Registration took {result['time']} seconds.")
    print(f"Reason for the registration algorithm to stop: {result['reason_registration_ended']}.")
    print("Relative norm of difference: "
          f"{np.linalg.norm(target - transformed_input) / np.linalg.norm(target)}")

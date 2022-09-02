import time
import pickle
import numpy as np

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction


if __name__ == "__main__":
    # define greyscale images
    N = 100
    input_ = ScalarFunction((N,))
    target = ScalarFunction((N,))
    input_[N//5:2*N//5] = 1
    target[2*N//5:3*N//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=10., exponent=3.)
    start = time.time()
    result = gs.register(input_, target, sigma=0.005, return_all=True)
    end = time.time()
    full_registration_time = end - start

    image = result['transformed_input']
    v0 = result['initial_vector_field']

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {(target - image).norm / target.norm}')

    rb = [v0 / v0.norm,]#.reshape((v0.flatten().shape[0], 1)) / np.linalg.norm(v0.flatten())
    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(rb, alpha=6., exponent=2)

    reduced_quantities = reduced_gs.get_reduced_quantities()

    with open('reduced_quantities_1d_example', 'wb') as file_obj:
        pickle.dump(reduced_quantities, file_obj)

    start = time.time()
    result = reduced_gs.register(input_, target, sigma=0.01, return_all=True)
    end = time.time()
    reduced_registration_time = end - start
    reduced_image = result['transformed_input']

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {reduced_image}')
    print(f'Relative norm of difference: {(target - reduced_image).norm / target.norm}')

    print()
    print("Computation times:")
    print(f"Full: {full_registration_time}")
    print(f"Reduced: {reduced_registration_time}")

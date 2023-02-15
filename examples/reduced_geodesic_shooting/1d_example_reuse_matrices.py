import time
import pickle

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction


if __name__ == "__main__":
    # define greyscale images
    N = 100
    template = ScalarFunction((N,))
    target = ScalarFunction((N,))
    template[N//5:2*N//5] = 1
    target[2*N//5:3*N//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=10., exponent=3)
    start = time.time()
    result = gs.register(template, target, sigma=0.005, return_all=True)
    end = time.time()
    full_registration_time = end - start

    image = result['transformed_input']
    v0 = result['initial_vector_field']

    print(f'Input: {template}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {(target - image).norm / target.norm}')

    with open('reduced_quantities_1d_example', 'rb') as file_obj:
        reduced_quantities = pickle.load(file_obj)

    rb = None
    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(rb, alpha=6., exponent=2,
                                                           precomputed_quantities=reduced_quantities)

    start = time.time()
    result_reduced = reduced_gs.register(template, target, sigma=0.01, return_all=True)
    end = time.time()
    reduced_registration_time = end - start
    reduced_image = result_reduced['transformed_input']

    print(f'Input: {template}')
    print(f'Target: {target}')
    print(f'Registration result: {reduced_image}')
    print(f'Relative norm of difference: {(target - reduced_image).norm / target.norm}')

    print()
    print("Computation times:")
    print(f"Full: {full_registration_time}")
    print(f"Reduced: {reduced_registration_time}")

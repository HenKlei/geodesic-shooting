import time
import pickle

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction, VectorField
from geodesic_shooting.utils.reduced import pod
from geodesic_shooting.utils.helper_functions import lincomb


if __name__ == "__main__":
    # define greyscale images
    N = 100
    input_ = ScalarFunction((N,))
    target = ScalarFunction((N,))
    input_[N//5:2*N//5] = 1
    target[2*N//5:3*N//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=10., exponent=3)
    start = time.time()
    result = gs.register(input_, target, sigma=0.005, return_all=True)
    end = time.time()
    full_registration_time = end - start

    image = result['transformed_input']

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {(target - image).norm / target.norm}')

    print()
    print("========== Reduced Geodesic Shooting ==========")
    avg = result['vector_fields'].average
    rb = [r - avg for r in result['vector_fields']]
    product_operator = gs.regularizer.cauchy_navier
    rb_size = 2
    rb, singular_values = pod(rb, num_modes=rb_size, product_operator=product_operator,
                              return_singular_values='all', shift=avg)
    print(f'Singular values: {singular_values}')
    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(rb, alpha=6., exponent=2)

    reduced_quantities = reduced_gs.get_reduced_quantities()

    with open('reduced_quantities_1d_example', 'wb') as file_obj:
        pickle.dump(reduced_quantities, file_obj)

    start = time.time()
    result_reduced = reduced_gs.register(input_, target, sigma=0.01, return_all=True)
    end = time.time()
    reduced_registration_time = end - start
    reduced_image = result_reduced['transformed_input']

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {reduced_image}')
    print(f'Relative norm of difference: {(target - reduced_image).norm / target.norm}')

    print()
    print("Computation times:")
    print(f"Full: {full_registration_time}")
    print(f"Reduced: {reduced_registration_time}")

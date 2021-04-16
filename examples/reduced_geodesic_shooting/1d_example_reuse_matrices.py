import pickle
import time
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
    gs = geodesic_shooting.GeodesicShooting(alpha=6., exponent=2)

    start = time.time()
    image, v0, energies, Phi0, length = gs.register(input_, target, sigma=0.01,
                                                    epsilon=0.001, return_all=True)
    end = time.time()
    full_registration_time = end - start

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {np.linalg.norm(target - image) / np.linalg.norm(target)}')

    with open('reduced_quantities_1d_example', 'rb') as file_obj:
        reduced_matrices = pickle.load(file_obj)

    rb = None
    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(input_.shape, rb, alpha=6., exponent=2,
                                                           precomputed_quantities=reduced_matrices)

    start = time.time()
    image, v0, energies, Phi0, length = reduced_gs.register(input_, target, sigma=0.01,
                                                            epsilon=0.001, return_all=True)
    end = time.time()
    reduced_registration_time = end - start

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Reduced registration result: {image}')
    print(f'Relative norm of difference: {np.linalg.norm(target - image) / np.linalg.norm(target)}')

    print()
    print("Computation times:")
    print(f"Full: {full_registration_time}")
    print(f"Reduced: {reduced_registration_time}")

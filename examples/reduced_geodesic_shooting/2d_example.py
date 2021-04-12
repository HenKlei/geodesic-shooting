import time
import numpy as np

import geodesic_shooting


if __name__ == "__main__":
    # define greyscale images
    N = 10
    M = 5
    input_ = np.zeros((N, M))
    target = np.zeros((N, M))
    input_[N//5:2*N//5, M//5:2*M//5] = 1
    target[2*N//5:3*N//5, M//5:2*M//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=6., exponent=3)

    start = time.time()
    image, v0, energies, Phi0, length = gs.register(input_, target, sigma=0.001,
                                                    epsilon=0.000005, return_all=True)
    end = time.time()
    full_registration_time = end - start

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {np.linalg.norm(target - image) / np.linalg.norm(target)}')

    rb = v0.reshape((v0.flatten().shape[0], 1)) / np.linalg.norm(v0.flatten())
    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(input_.shape, rb, alpha=6., exponent=3)

    start = time.time()
    image, v0, energies, Phi0, length = reduced_gs.register(input_, target, sigma=0.001,
                                                            epsilon=0.000005, return_all=True)
    end = time.time()
    reduced_registration_time = end - start

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {np.linalg.norm(target - image) / np.linalg.norm(target)}')

    print()
    print("Computation times:")
    print(f"Full: {full_registration_time}")
    print(f"Reduced: {reduced_registration_time}")

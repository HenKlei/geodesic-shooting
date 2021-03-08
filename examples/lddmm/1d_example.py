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
    lddmm = geodesic_shooting.LDDMM(alpha=10., gamma=1.)
    image, v, energies, length, Phi0, Phi1, J0, J1 = lddmm.register(input_, target, sigma=0.05, epsilon=0.01, return_all=True)

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {np.linalg.norm(target - image) / np.linalg.norm(target)}')

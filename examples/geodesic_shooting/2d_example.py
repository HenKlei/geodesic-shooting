import numpy as np

import pyLDDMM


if __name__ == "__main__":
    # define greyscale images
    N = 10
    M = 5
    input_ = np.zeros((N, M))
    target = np.zeros((N, M))
    input_[N//5:2*N//5, M//5:2*M//5] = 1
    target[2*N//5:3*N//5, M//5:2*M//5] = 1

    problem = pyLDDMM.ImageRegistrationProblemGS(target, alpha=6, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.GeodesicShooting()
    image, v0, energies, Phi0, length = lddmm.register(input_, problem, sigma=0.01, epsilon=0.0001, return_all=True)

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {np.linalg.norm(target - image) / np.linalg.norm(target)}')

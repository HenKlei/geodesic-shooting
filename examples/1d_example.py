import numpy as np

import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid


if __name__ == "__main__":
    # define greyscale images
    N = 100
    input_ = np.zeros(N)
    target = np.zeros(N)
    input_[N//5:2*N//5] = 1
    target[2*N//5:3*N//5] = 1

    problem = pyLDDMM.ImageRegistrationProblem(target, alpha=10, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.LDDMM()
    image, v, energies, length, Phi0, Phi1, J0, J1 = lddmm.register(input_, problem, sigma=0.05, epsilon=0.01, return_all=True)

    print(f'Input: {input_}')
    print(f'Target: {target}')
    print(f'Registration result: {image}')
    print(f'Relative norm of difference: {np.linalg.norm(target - image) / np.linalg.norm(target)}')

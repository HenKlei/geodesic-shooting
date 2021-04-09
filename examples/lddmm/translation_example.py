import os
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.io import save_image, save_animation
from geodesic_shooting.utils.visualization import plot_warpgrid
from geodesic_shooting.utils.create_example_images import make_circle


if __name__ == "__main__":
    # create images
    input_ = (make_circle(64, np.array([25, 40]), 18) * 0.2
              + make_circle(64, np.array([25, 40]), 15) * 0.8)
    target = (make_circle(64, np.array([40, 25]), 18) * 0.2
              + make_circle(64, np.array([40, 25]), 15) * 0.8)

    # perform the registration
    lddmm = geodesic_shooting.LDDMM(alpha=1000., gamma=1.)
    image, v, energies, Phi0, Phi1, J0, J1, length = lddmm.register(input_, target, sigma=0.1,
                                                                    epsilon=0.01, iterations=15,
                                                                    return_all=True)

    norm = np.linalg.norm((target - image).flatten()) / np.linalg.norm(target.flatten())
    print(f'Relative norm of difference: {norm}')

    plt.matshow(input_)
    plt.title("Input")
    plt.matshow(target)
    plt.title("Target")
    plt.matshow(image)
    plt.title("Result")

    FILEPATH_RESULTS = 'results/'
    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # save input_ aligned to target
    save_image(image, FILEPATH_RESULTS + 'translation.png')

    # save animation of the transformation
    save_animation(J0, FILEPATH_RESULTS + 'translation.gif')

    # plot the transformation
    fig = plot_warpgrid(Phi1[0], title="Warp grid (translation)", interval=2)
    fig.savefig(FILEPATH_RESULTS + 'translation_warp.png')

    plt.show()

import os
import matplotlib.pyplot as plt

import pyLDDMM
from pyLDDMM.utils.io import load_image, save_image, save_animation
from pyLDDMM.utils.visualization import plot_warpgrid, plot_vector_field


if __name__ == "__main__":
    # load greyscale images
    input_ = load_image('../example_images/circle.png')
    target = load_image('../example_images/square.png')

    problem = pyLDDMM.ImageRegistrationProblem(target, alpha=1, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.LDDMM()
    image, v, energies, Phi0, Phi1, J0, J1, length = lddmm.register(input_, problem, sigma=0.1, epsilon=0.0001, K=50, return_all=True)

    FILEPATH_RESULTS = 'results/'

    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # save input_ aligned to target
    save_image(image, FILEPATH_RESULTS + 'circle_to_square.png')

    # save animation of the transformation
    save_animation(J0, FILEPATH_RESULTS + 'circle_to_square.gif')

    # plot the transformation
    fig = plot_warpgrid(Phi1[0], title="Warp grid (C2S)", interval=1)
    fig.savefig(FILEPATH_RESULTS + 'circle_to_square_warp.png')

    # plot the (initial) vector field
    plot_vector_field(v[0], title="Initial vector field (C2S)", interval=2)

    # plot the deformation vector field
    plot_vector_field(Phi0[0] - Phi0[-1], title="Overall deformation vector field (C2S)", interval=2)

    plt.show()

import os

import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid, plot_vector_field


if __name__ == "__main__":
    # load greyscale images
    input_ = loadimg('../example_images/square.png')
    target = loadimg('../example_images/circle.png')

    problem = pyLDDMM.ImageRegistrationProblem(target, alpha=1, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.LDDMM()
    im, v, energies, Phi0, Phi1, J0, J1, length = lddmm.register(input_, problem, sigma=0.1, epsilon=0.0001, K=50, return_all=True)

    FILEPATH_RESULTS = 'results/'

    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # save input_ aligned to target
    saveimg(FILEPATH_RESULTS + 'circle_to_square.png', im)

    # save animation of the transformation
    save_animation(FILEPATH_RESULTS + 'circle_to_square.gif', J0)

    # plot the transformation
    plt = plot_warpgrid(Phi1[0], title="Warp grid (C2S)", interval=1)
    plt.savefig(FILEPATH_RESULTS + 'circle_to_square_warp.png')

    # plot the (initial) vector field
    plt = plot_vector_field(v[0], title="Initial vector field (C2S)", interval=2)
    plt.show()

    # plot the deformation vector field
    plt = plot_vector_field(Phi0[0] - Phi0[-1], title="Overall deformation vector field (C2S)", interval=2)
    plt.show()

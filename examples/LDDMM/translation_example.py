import os

import pyLDDMM
from pyLDDMM.utils.io import load_image, save_image, save_animation
from pyLDDMM.utils.visualization import plot_warpgrid


if __name__ == "__main__":
    # load greyscale images
    input_ = load_image('../example_images/translation_input.png')
    target = load_image('../example_images/translation_target.png')

    problem = pyLDDMM.ImageRegistrationProblem(target, alpha=1000, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.LDDMM()
    image, v, energies, Phi0, Phi1, J0, J1, length = lddmm.register(input_, problem, sigma=0.1, epsilon=0.01, K=15, return_all=True)

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

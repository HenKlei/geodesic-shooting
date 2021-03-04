import os

import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid

if __name__ == "__main__":
    # load greyscale images
    input_ = loadimg('../example_images/translation_input.png')
    target = loadimg('../example_images/translation_target.png')

    problem = pyLDDMM.ImageRegistrationProblem(target, alpha=1000, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.LDDMM()
    im, v, energies, Phi0, Phi1, J0, J1, length = lddmm.register(input_, problem, sigma=0.1, epsilon=0.01, K=15, return_all=True)

    FILEPATH_RESULTS = 'results/'
  
    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # save input_ aligned to target
    saveimg(FILEPATH_RESULTS + 'translation.png', im)

    # save animation of the transformation
    save_animation(FILEPATH_RESULTS + 'translation.gif', J0)

    # plot the transformation
    plt = plot_warpgrid(Phi1[0], title="Warp grid (translation)", interval=2)
    plt.savefig(FILEPATH_RESULTS + 'translation_warp.png')

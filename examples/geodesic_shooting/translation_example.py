import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid

if __name__ == "__main__":
    # load greyscale images
    i0 = loadimg('../example_images/translation_input.png')
    i1 = loadimg('../example_images/translation_target.png')

    problem = pyLDDMM.ImageRegistrationProblemGS(i1, alpha=1000, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.GeodesicShooting()
    im, v0, energies, Phi0, length = lddmm.register(i0, problem, sigma=0.1, epsilon=0.01, K=50, return_all=True)

    # save i0 aligned to i1
    saveimg('../example_images/out_translation.png', im)

    plt_inverse = plot_warpgrid(Phi0, interval=2)
    plt_inverse.savefig('../example_images/out_translation_warp_inverse.png')

import os

import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid, plot_vector_field


if __name__ == "__main__":
    # load greyscale images
    input_ = loadimg('../example_images/square.png')
    target = loadimg('../example_images/circle.png')

    problem = pyLDDMM.ImageRegistrationProblemGS(target, alpha=6, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.GeodesicShooting()
    im, v0, energies, Phi0, length = lddmm.register(input_, problem, sigma=0.1, epsilon=0.01, K=50, return_all=True)

    FILEPATH_RESULTS = 'results/'

    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # save input_ aligned to target
    saveimg(FILEPATH_RESULTS + 'square_to_circle.png', im)

    # plot the inverse transformation
    plt_inverse = plot_warpgrid(Phi0, title="Inverse warp grid (S2C)", interval=2)
    plt_inverse.savefig(FILEPATH_RESULTS + 'square_to_circle_warp_inverse.png')

    # multiply initial vector field by 0.5, integrate it forward and push the input_ image along this flow
    Phi_half = lddmm.integrate_forward_flow(lddmm.integrate_forward_vector_field(v0 / 2.))
    saveimg(FILEPATH_RESULTS + 'square_to_circle_half_speed.png', lddmm.push_forward(input_, Phi_half))

    # plot the (initial) vector field
    plt = plot_vector_field(v0, title="Initial vector field (S2C)", interval=2)
    plt.show()

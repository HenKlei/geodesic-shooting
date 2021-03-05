import os
import matplotlib.pyplot as plt

import pyLDDMM
from pyLDDMM.utils.io import load_image, save_image, save_animation
from pyLDDMM.utils.visualization import plot_warpgrid, plot_vector_field


if __name__ == "__main__":
    # load greyscale images
    input_ = load_image('../example_images/circle.png')
    target = load_image('../example_images/square.png')

    problem = pyLDDMM.ImageRegistrationProblemGS(target, alpha=6, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.GeodesicShooting()
    image, v0, energies, Phi0, length = lddmm.register(input_, problem, sigma=0.1, epsilon=0.01, K=50, return_all=True)

    FILEPATH_RESULTS = 'results/'

    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # save input_ aligned to target
    save_image(image, FILEPATH_RESULTS + 'circle_to_square.png')

    # plot the inverse transformation
    fig_inverse = plot_warpgrid(Phi0, title="Inverse warp grid (C2S)", interval=2)
    fig_inverse.savefig(FILEPATH_RESULTS + 'circle_to_square_warp_inverse.png')

    # multiply initial vector field by 0.5, integrate it forward and push the input_ image along this flow
    Phi_half = lddmm.integrate_forward_flow(lddmm.integrate_forward_vector_field(v0 / 2.))
    save_image(lddmm.push_forward(input_, Phi_half), FILEPATH_RESULTS + 'circle_to_square_half_speed.png')

    # plot the (initial) vector field
    plot_vector_field(v0, title="Initial vector field (C2S)", interval=2)

    plt.show()

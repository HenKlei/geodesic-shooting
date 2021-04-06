import os
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.io import load_image, save_image
from geodesic_shooting.utils.visualization import plot_warpgrid, plot_vector_field


if __name__ == "__main__":
    # load greyscale images
    input_ = load_image('../example_images/translation_input.png')
    target = load_image('../example_images/translation_target.png')

    # perform the registration
    gs = geodesic_shooting.TestGeodesicShooting(alpha=10., exponent=4)
    image, v0, energies, Phi0, length = gs.register(input_, target, sigma=0.01,
                                                    epsilon=0.001, iterations=100,
                                                    return_all=True)

    FILEPATH_RESULTS = 'results/'
    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # save input_ aligned to target
    save_image(image, FILEPATH_RESULTS + 'translation.png')

    # plot the inverse transformation
    fig_inverse = plot_warpgrid(Phi0, title="Inverse warp grid (translation)", interval=2)
    fig_inverse.savefig(FILEPATH_RESULTS + 'translation_warp_inverse.png')

    # multiply initial vector field by 0.5, integrate it forward and
    # push the input_ image along this flow
    Phi_half = gs.integrate_forward_flow(gs.integrate_forward_vector_field(v0 / 2.))
    save_image(gs.push_forward(input_, Phi_half),
               FILEPATH_RESULTS + 'translation_half_speed.png')

    # plot the (initial) vector field
    plot_vector_field(v0, title="Initial vector field (translation)", interval=2)

    plt.show()

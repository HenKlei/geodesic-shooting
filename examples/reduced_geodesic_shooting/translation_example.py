import os
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.io import save_image
from geodesic_shooting.utils.visualization import plot_warpgrid, plot_vector_field
from geodesic_shooting.utils.create_example_images import make_circle


if __name__ == "__main__":
    # create images
    input_ = (make_circle(64, np.array([25, 40]), 18) * 0.2
              + make_circle(64, np.array([25, 40]), 15) * 0.8)
    target = (make_circle(64, np.array([40, 25]), 18) * 0.2
              + make_circle(64, np.array([40, 25]), 15) * 0.8)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=1000., exponent=3)
    image, v0, energies, Phi0, length = gs.register(input_, target, sigma=0.1,
                                                    epsilon=0.1, iterations=20,
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

    rb = v0.reshape((v0.flatten().shape[0], 1)) / np.linalg.norm(v0.flatten())
    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(rb, input_.shape, alpha=1000.,
                                                           exponent=3)
    image, v0, energies, Phi0, length = reduced_gs.register(input_, target, sigma=0.1,
                                                            epsilon=0.1, iterations=20,
                                                            return_all=True)

    plt.matshow(image)
    plt.title("Result reduced")

    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.io import save_image
from geodesic_shooting.utils.visualization import plot_warpgrid, plot_vector_field
from geodesic_shooting.utils.create_example_images import make_circle, make_square


if __name__ == "__main__":
    # create images
    target = make_circle(64, np.array([32, 32]), 20)
    input_ = make_square(64, np.array([32, 32]), 40)

    # perform the registration
    geodesic_shooting = geodesic_shooting.GeodesicShooting(alpha=6., exponent=1)
    image, v0, energies, Phi0, length = geodesic_shooting.register(input_, target, sigma=0.1,
                                                                   epsilon=0.01, iterations=100,
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
    save_image(image, FILEPATH_RESULTS + 'square_to_circle.png')

    # plot the inverse transformation
    fig_inverse = plot_warpgrid(Phi0, title="Inverse warp grid (S2C)", interval=2)
    fig_inverse.savefig(FILEPATH_RESULTS + 'square_to_circle_warp_inverse.png')

    # multiply initial vector field by 0.5, integrate it forward and
    # push the input_ image along this flow
    Phi_half = geodesic_shooting.integrate_forward_flow(
        geodesic_shooting.integrate_forward_vector_field(v0 / 2.))
    save_image(geodesic_shooting.push_forward(input_, Phi_half),
               FILEPATH_RESULTS + 'square_to_circle_half_speed.png')

    # plot the (initial) vector field
    plot_vector_field(v0, title="Initial vector field (S2C)", interval=2)

    plt.show()

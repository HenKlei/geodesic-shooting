import os
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.io import save_image, save_animation
from geodesic_shooting.utils.visualization import plot_warpgrid, plot_vector_field
from geodesic_shooting.utils.create_example_images import make_circle, make_square


if __name__ == "__main__":
    # create images
    target = make_circle(64, np.array([32, 32]), 20)
    input_ = make_square(64, np.array([32, 32]), 40)

    # perform the registration
    lddmm = geodesic_shooting.LDDMM(alpha=1., exponent=1.)
    image, v, energies, Phi0, Phi1, J0, J1, length = lddmm.register(input_, target, sigma=0.1,
                                                                    epsilon=0.0001, iterations=100,
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

    # save animation of the transformation
    save_animation(J0, FILEPATH_RESULTS + 'square_to_circle.gif')

    # plot the transformation
    fig = plot_warpgrid(Phi1[0], title="Warp grid (S2C)", interval=1)
    fig.savefig(FILEPATH_RESULTS + 'square_to_circle_warp.png')

    # plot the (initial) vector field
    plot_vector_field(v[0], title="Initial vector field (S2C)", interval=2)

    # plot the deformation vector field
    plot_vector_field(Phi0[0] - Phi0[-1], title="Overall deformation vector field (S2C)",
                      interval=2)

    plt.show()

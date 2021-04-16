import os
import pickle
import time
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

    start = time.time()
    image, v0, energies, Phi0, length = gs.register(input_, target, sigma=0.1,
                                                    epsilon=0.1, iterations=20,
                                                    return_all=True)
    end = time.time()
    full_registration_time = end - start

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

    with open('reduced_quantities_translation_example', 'rb') as file_obj:
        reduced_matrices = pickle.load(file_obj)

    rb = None
    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(input_.shape, rb, alpha=1000.,
                                                           exponent=3,
                                                           precomputed_quantities=reduced_matrices)

    start = time.time()
    image, v0, energies, Phi0, length = reduced_gs.register(input_, target, sigma=0.1,
                                                            epsilon=0.1, iterations=20,
                                                            return_all=True)
    end = time.time()
    reduced_registration_time = end - start

    norm = np.linalg.norm((target - image).flatten()) / np.linalg.norm(target.flatten())
    print(f'Relative norm of difference: {norm}')

    plt.matshow(image)
    plt.title("Result reduced")

    plt.show()

    print()
    print("Computation times:")
    print(f"Full: {full_registration_time}")
    print(f"Reduced: {reduced_registration_time}")

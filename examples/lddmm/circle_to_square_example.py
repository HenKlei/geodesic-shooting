import os
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.io import save_image, save_animation
from geodesic_shooting.utils.visualization import plot_warpgrid, plot_vector_field
from geodesic_shooting.utils.create_example_images import make_circle, make_square


if __name__ == "__main__":
    # create images
    input_ = make_circle(64, np.array([32, 32]), 20)
    target = make_square(64, np.array([32, 32]), 40)

    # perform the registration
    lddmm = geodesic_shooting.LDDMM(alpha=1., exponent=1.)
    result = lddmm.register(input_, target, sigma=0.1, epsilon=0.0001, iterations=50,
                            return_all=True)

    transformed_input = result['transformed_input']
    J0 = result['forward_pushed_input']
    Phi0 = result['forward_flows']
    Phi1 = result['backward_flows']
    v = result['velocity_fields']

    print(f"Registration finished after {result['iterations']} iterations.")
    print(f"Registration took {result['time']} seconds.")
    print(f"Reason for the registration algorithm to stop: {result['reason_registration_ended']}.")

    norm = np.linalg.norm((target - transformed_input).flatten()) / np.linalg.norm(target.flatten())
    print(f'Relative norm of difference: {norm}')

    plt.matshow(input_)
    plt.title("Input")
    plt.matshow(target)
    plt.title("Target")
    plt.matshow(transformed_input)
    plt.title("Result")

    FILEPATH_RESULTS = 'results/'
    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # save input_ aligned to target
    save_image(transformed_input, FILEPATH_RESULTS + 'circle_to_square.png')

    # save animation of the transformation
    save_animation(J0, FILEPATH_RESULTS + 'circle_to_square.gif')

    # plot the transformation
    fig = plot_warpgrid(Phi1[0], title="Warp grid (C2S)", interval=1)
    fig.savefig(FILEPATH_RESULTS + 'circle_to_square_warp.png')

    # plot the (initial) vector field
    plot_vector_field(v[0], title="Initial vector field (C2S)", interval=2)

    # plot the deformation vector field
    plot_vector_field(Phi0[0] - Phi0[-1], title="Overall deformation vector field (C2S)",
                      interval=2)

    plt.show()

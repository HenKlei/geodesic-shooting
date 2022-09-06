import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle, make_square
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    input_ = make_circle((64, 64), np.array([32, 32]), 20)
    target = make_square((64, 64), np.array([32, 32]), 40)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=6., exponent=2)
    result = gs.register(input_, target, sigma=0.1, return_all=True)

    plot_registration_results(result)
    save_plots_registration_results(result, filepath='results_circle_to_square/')

import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_square
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    target = make_square((64, 64), np.array([50, 50]), 30)
    template = make_square((64, 64), np.array([32, 32]), 30)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=100., exponent=3)
    result = gs.register(template, target, sigma=0.1, return_all=True)

    plot_registration_results(result, frequency=5)
    save_plots_registration_results(result, filepath='results_moving_square/')

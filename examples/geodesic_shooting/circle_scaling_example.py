import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    template = make_circle((64, 64), np.array([32, 32]), 10)
    target = make_circle((64, 64), np.array([32, 32]), 20)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=20., exponent=3)
    result = gs.register(template, target, sigma=0.01, return_all=True)

    plot_registration_results(result)
    save_plots_registration_results(result, filepath='results_circle_scaling/')
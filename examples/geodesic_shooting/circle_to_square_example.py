import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle, make_square
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    template = make_circle((64, 64), np.array([32, 32]), 20)
    target = make_square((64, 64), np.array([32, 32]), 40)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.1, exponent=1, gamma=5.)
    result = gs.register(template, target, sigma=0.01, return_all=True, optimization_method='GD',
                         optimizer_options={'maxiter': 50, 'grad_norm_tol': 1e-3})

    plot_registration_results(result, frequency=5)
    save_plots_registration_results(result, filepath='results_circle_to_square/')

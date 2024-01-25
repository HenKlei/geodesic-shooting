import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle, make_square
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    target = make_circle((64, 64), np.array([32, 32]), 20)
    template = make_square((64, 64), np.array([32, 32]), 40)

    # set restriction of where to compute the error and the gradient
    restriction = np.s_[2:-20, 2:-20]

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.01, exponent=1)
    result = gs.register(template, target, sigma=0.01, return_all=True, restriction=restriction,
                         optimization_method='GD', optimizer_options={'maxiter': 20})

    result['initial_vector_field'].save_tikz('initial_vector_field_square_to_circle.tex',
                                             title="Initial vector field square to circle",
                                             interval=2, scale=100)

    plot_registration_results(result, frequency=5)
    save_plots_registration_results(result, filepath='results_square_to_circle/', save_animations=True)

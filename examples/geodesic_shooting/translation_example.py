import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    template = (make_circle((64, 64), np.array([25, 35]), 15) * 0.2
              + make_circle((64, 64), np.array([25, 35]), 12) * 0.8)
    target = (make_circle((64, 64), np.array([35, 25]), 15) * 0.2
              + make_circle((64, 64), np.array([35, 25]), 12) * 0.8)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.01, exponent=2, gamma=1.)
    result = gs.register(template, target, sigma=0.01, return_all=True, optimization_method='GD')

    result['initial_vector_field'].save_tikz('initial_vector_field_translation.tex',
                                             title="Initial vector field translation",
                                             interval=5, scale=10)

    plot_registration_results(result)
    save_plots_registration_results(result, filepath='results_translation/')

import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    input_ = (make_circle((64, 64), np.array([25, 40]), 18) * 0.2
              + make_circle((64, 64), np.array([25, 40]), 15) * 0.8)
    target = (make_circle((64, 64), np.array([40, 25]), 18) * 0.2
              + make_circle((64, 64), np.array([40, 25]), 15) * 0.8)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=100., exponent=2.)
    result = gs.register(input_, target, sigma=0.1, return_all=True)

    plot_registration_results(result)
    save_plots_registration_results(result, filepath='results_translation/')

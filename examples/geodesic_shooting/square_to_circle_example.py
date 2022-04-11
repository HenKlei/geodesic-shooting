import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle, make_square
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    target = make_circle((64, 64), np.array([32, 32]), 20)
    input_ = make_square((64, 64), np.array([32, 32]), 40)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=10., exponent=4.)
    result = gs.register(input_, target, sigma=0.01, return_all=True)

    result['initial_vector_field'].save_tikz('initial_vector_field_square_to_circle.tex',
                                             title="Initial vector field square to circle",
                                             interval=5, scale=10)

    plot_registration_results(result)
    save_plots_registration_results(result, filepath='results_square_to_circle/')

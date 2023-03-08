import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_square
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    target = make_square((100, 100), np.array([75, 75]), 30)
    template = make_square((100, 100), np.array([55, 55]), 30)

    target.plot("Target")
    template.plot("Template")
    import matplotlib.pyplot as plt
    plt.show()

    restriction = np.s_[20:-20, 20:-20]

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=300., exponent=3)
    result = gs.register(template, target, sigma=0.01, return_all=True, restriction=restriction)

    plot_registration_results(result, frequency=5)
    save_plots_registration_results(result, filepath='results_moving_square/')

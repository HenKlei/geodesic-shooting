import pickle
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.reduced import pod
from geodesic_shooting.utils.create_example_images import make_circle


if __name__ == "__main__":
    # create images
    input_ = (make_circle(64, np.array([25, 40]), 18) * 0.2
              + make_circle(64, np.array([25, 40]), 15) * 0.8)
    target = (make_circle(64, np.array([40, 25]), 18) * 0.2
              + make_circle(64, np.array([40, 25]), 15) * 0.8)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=1000., exponent=3)

    velocity_fields = []
    for x in range(0, 25, 5):
        target = (make_circle(64, np.array([25+x, 40-x]), 18) * 0.2
                  + make_circle(64, np.array([25+x, 40-x]), 15) * 0.8)
        plt.matshow(target)
        plt.show()
        image, v0, energies, Phi0, length = gs.register(input_, target, sigma=0.1,
                                                        epsilon=0.1, iterations=20,
                                                        return_all=True)
        plt.matshow(image-target)
        plt.show()
        velocity_fields.append(v0.flatten())

    velocity_fields = np.stack(velocity_fields, axis=-1)
    rb = pod(velocity_fields, modes=2)

    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(input_.shape, rb, alpha=1000.,
                                                           exponent=3)

    reduced_quantities = reduced_gs.get_reduced_quantities()

    with open(f'reduced_quantities_translation_rb_size_{reduced_gs.rb_size}', 'wb') as file_obj:
        pickle.dump(reduced_quantities, file_obj)

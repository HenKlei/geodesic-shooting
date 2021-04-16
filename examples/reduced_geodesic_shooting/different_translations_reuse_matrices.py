import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.create_example_images import make_circle
from geodesic_shooting.utils.visualization import plot_vector_field


if __name__ == "__main__":
    # create images
    input_ = (make_circle(64, np.array([25, 40]), 18) * 0.2
              + make_circle(64, np.array([25, 40]), 15) * 0.8)
    target = (make_circle(64, np.array([35, 30]), 18) * 0.2
              + make_circle(64, np.array([35, 30]), 15) * 0.8)

    rb_size = 1
    rb = None
    with open(f'reduced_quantities_translation_rb_size_{rb_size}', 'rb') as file_obj:
        reduced_matrices = pickle.load(file_obj)

    plot_vector_field(reduced_matrices['rb_velocity_fields'][:, 0].reshape(2, 64, 64),
                      title="Initial vector field (translation)", interval=2)
    plt.show()

    reduced_gs = geodesic_shooting.ReducedGeodesicShooting(input_.shape, rb, alpha=1000.,
                                                           exponent=3,
                                                           precomputed_quantities=reduced_matrices)

    start = time.time()
    image, v0, energies, Phi0, length = reduced_gs.register(input_, target, sigma=0.1,
                                                            epsilon=0.1, iterations=10000,
                                                            return_all=True)
    end = time.time()
    reduced_registration_time = end - start

    norm = np.linalg.norm((target - image).flatten()) / np.linalg.norm(target.flatten())
    print(f'Relative norm of difference: {norm}')

    plt.matshow(input_)
    plt.title("Input")
    plt.matshow(target)
    plt.title("Target")
    plt.matshow(image)
    plt.title("Result reduced")

    plt.show()

    print()
    print("Computation times:")
    print(f"Reduced: {reduced_registration_time}")

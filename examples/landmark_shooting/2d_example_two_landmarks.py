import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.kernels import GaussianKernel

from geodesic_shooting.utils.visualization import (animate_landmark_trajectories,
                                                   plot_initial_momenta_and_landmarks,
                                                   plot_landmark_matchings,
                                                   plot_landmark_trajectories)
from geodesic_shooting.core import ScalarFunction


if __name__ == "__main__":
    # define landmark positions
    gs = geodesic_shooting.LandmarkShooting(kernel=GaussianKernel, kwargs_kernel={'sigma': 1.5},
                                            dim=2, num_landmarks=2)
    kernel = gs.kernel
    input_landmarks = np.array([[0., 0.], [10., 5.]])
    true_momenta = np.array([[5., 2.5], [-5., -2.5]])
    time_evolution_momenta, time_evolution_positions = gs.integrate_forward_Hamiltonian(true_momenta, input_landmarks)
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=gs.kernel,
                               show_vector_fields=False, title="True trajectories")
    plt.show()
    target_landmarks = time_evolution_positions[-1].reshape(input_landmarks.shape)

    sigma = 0.02
    # perform the registration using landmark shooting algorithm
    result = gs.register(input_landmarks, target_landmarks, optimization_method='GD',
                         sigma=sigma, return_all=True, landmarks_labeled=True,
                         initial_momenta=np.zeros_like(input_landmarks),
                         optimizer_options={'disp': True, 'maxiter': 100})
    final_momenta = result['initial_momenta']
    registered_landmarks = result['registered_landmarks']

    hamiltonian = 0.
    for pa, qa in zip(final_momenta, input_landmarks):
        for pb, qb in zip(final_momenta, input_landmarks):
            hamiltonian += 0.5 * pa @ kernel(qa, qb) @ pb
    assert np.isclose(result['energy_regularizer'], hamiltonian)

    print(f"Gradient: {result['grad']}")
    print(f"True momenta: {true_momenta}")

    # Check energy of true momenta
    energy, energy_regularizer, energy_intensity_unscaled, energy_intensity = gs.energy_and_gradient(
        true_momenta, input_landmarks, target_landmarks, sigma,
        lambda positions: np.linalg.norm(positions.flatten() - target_landmarks.flatten())**2,
        compute_grad=False, return_all_energies=True)
    print("True momenta energies:")
    print(f"Energy regularizer: {energy_regularizer}")
    print(f"Energy intensity unscaled: {energy_intensity_unscaled}")
    print(f"Energy intensity: {energy_intensity}")
    print(f"Energy: {energy}")

    print("Computed momenta energies:")
    print(f"Energy regularizer: {result['energy_regularizer']}")
    print(f"Energy intensity unscaled: {result['energy_intensity_unscaled']}")
    print(f"Energy intensity: {result['energy_intensity']}")
    print(f"Energy: {result['energy']}")

    print(f"Final momenta: {final_momenta}")
    print(f"Target landmarks: {target_landmarks}")
    print(f"Registered landmarks: {registered_landmarks}")
    print(f"Average distance: {np.average(np.linalg.norm(target_landmarks - registered_landmarks, axis=-1))}")

    time_integration_hamiltonian = 0.
    num_timesteps = result['time_evolution_momenta'].shape[0]
    for ps, qs in zip(result['time_evolution_momenta'], result['time_evolution_positions']):
        for pa, qa in zip(ps.reshape(input_landmarks.shape), qs.reshape(input_landmarks.shape)):
            for pb, qb in zip(ps.reshape(input_landmarks.shape), qs.reshape(input_landmarks.shape)):
                time_integration_hamiltonian += 0.5 * pa @ kernel(qa, qb) @ pb / num_timesteps

    assert np.abs(time_integration_hamiltonian - hamiltonian) / hamiltonian < 1e-2
    assert np.average(np.linalg.norm(final_momenta - true_momenta, axis=-1)) < 1e-2

    vf = gs.get_vector_field(final_momenta, result["input_landmarks"])
    vf.plot("Vector field at initial time", color_length=True)
    vf.get_magnitude().plot("Magnitude of vector field at initial time")

    # plot results
    plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks)
    plt.show()

    plot_initial_momenta_and_landmarks(final_momenta, registered_landmarks, kernel=kernel)

    time_evolution_momenta = result['time_evolution_momenta']
    time_evolution_positions = result['time_evolution_positions']
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=kernel)

    ani = animate_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=kernel)

    nx = 70
    ny = 60
    spatial_shape = (nx, ny)
    flow = gs.compute_diffeomorphism(final_momenta, input_landmarks, spatial_shape=spatial_shape)
    flow.plot("Flow")

    def set_landmarks_in_image(img, landmarks, sigma=1.):
        x, y = np.meshgrid(np.linspace(0., 1., nx), np.linspace(0., 1., ny))
        for i, l in enumerate(landmarks):
            dst = (x - l[0])**2 + (y - l[1])**2
            img += (i + 1.) * np.exp(-(dst.T / (sigma**2))) / (sigma**2)

    sigma = kernel.sigma
    image = ScalarFunction(spatial_shape)
    target_image = ScalarFunction(spatial_shape)
    set_landmarks_in_image(image, input_landmarks, sigma=sigma)
    set_landmarks_in_image(target_image, target_landmarks, sigma=sigma)

    image.plot("Original image")
    target_image.plot("Target image")
    resulting_image = image.push_forward(flow)
    resulting_image.plot("Transformed image")
    plt.show()

    print(f"Input: {input_landmarks}")
    print(f"Target: {target_landmarks}")
    print(f"Result: {registered_landmarks}")
    error = np.linalg.norm(target_landmarks - registered_landmarks)
    print(f"Norm of difference: {error}")

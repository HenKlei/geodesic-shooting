# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting

from geodesic_shooting.utils.visualization import (animate_landmark_trajectories,
                                                   plot_initial_momenta_and_landmarks,
                                                   plot_landmark_matchings,
                                                   plot_landmark_trajectories, animate_warpgrids)
from geodesic_shooting.core import ScalarFunction


# +
# define landmark positions
input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])

# perform the registration using landmark shooting algorithm
gs = geodesic_shooting.LandmarkShooting()
result = gs.register(input_landmarks, target_landmarks, sigma=0.1, return_all=True)
final_momenta = result['initial_momenta']
registered_landmarks = result['registered_landmarks']
# -

from geodesic_shooting.core import TimeDependentVectorField, VectorField
from geodesic_shooting.utils.visualization import construct_vector_field
from geodesic_shooting.utils import grid
from geodesic_shooting.utils import sampler


# +
def get_vector_field(momenta, positions,
                     mins=np.array([0., 0.]), maxs=np.array([1., 1.]),
                     spatial_shape=(100, 100)):

    """Evaluates vector field given by positions and momenta at grid points.

    Parameters
    ----------
    momenta
        Array containing the momenta of the landmarks.
    positions
        Array containing the positions of the landmarks.

    Returns
    -------
    Vector field at the grid points.
    """
    vector_field = VectorField(spatial_shape)

    vf_func = construct_vector_field(momenta.reshape((-1, gs.dim)),
                                     positions.reshape((-1, gs.dim)),
                                     kernel=gs.kernel)

    for pos in np.ndindex(spatial_shape):
        spatial_pos = mins + (maxs - mins) / np.array(spatial_shape) * np.array(pos)
        vector_field[pos] = vf_func(spatial_pos) * np.array(spatial_shape)

    return vector_field

def compute_time_evolution_of_diffeomorphisms(initial_momenta, initial_positions,
                                              mins=np.array([0., 0.]), maxs=np.array([1., 1.]),
                                              spatial_shape=(100, 100)):
    """Performs forward integration of diffeomorphism on given grid using the given
       initial momenta and positions.

    Parameters
    ----------
    initial_momenta
        Array containing the initial momenta of the landmarks.
    initial_positions
        Array containing the initial positions of the landmarks.
    grid
        Array containing the grid points.

    Returns
    -------
    Array containing the diffeomorphism at the different time instances.
    """
    assert mins.ndim == 1 and mins.shape[0] == len(spatial_shape)
    assert maxs.ndim == 1 and maxs.shape[0] == len(spatial_shape)
    assert np.all(mins < maxs)
    assert initial_momenta.shape == initial_positions.shape

    momenta, positions = gs.integrate_forward_Hamiltonian(initial_momenta.flatten(), initial_positions.flatten())
    vector_fields = TimeDependentVectorField(spatial_shape, gs.time_steps)

    for t in range(gs.time_steps - 1):
        vector_fields[t] = get_vector_field(momenta[t], positions[t], mins, maxs, spatial_shape)

    flow = integrate_forward_flow(vector_fields, spatial_shape)

    return flow

def integrate_forward_flow(vector_fields, spatial_shape):
    """Computes forward integration according to given vector fields.

    Parameters
    ----------
    vector_fields
        Sequence of vector fields (i.e. time-depending vector field).

    Returns
    -------
    Array containing the flow at the final time.
    """
    # make identity grid
    identity_grid = grid.coordinate_grid(spatial_shape)

    # initial flow is the identity mapping
    flow = identity_grid.copy()

    # perform forward integration
    for t in range(0, gs.time_steps-1):
        #flow -= gs.dt * sampler.sample(vector_fields[t], flow)
        flow = sampler.sample(flow, identity_grid + gs.dt * vector_fields[t])
        flow.plot(f"t={t}")

    return flow

def push_forward(image, flow):
    """Pushes forward an image along a flow.

    Parameters
    ----------
    image
        `ScalarFunction` to push forward.
    flow
        `VectorField` containing the flow according to which to push the input forward.

    Returns
    -------
    Array with the forward-pushed image.
    """
    return sampler.sample(image, flow)


# -

plt.rcParams['figure.figsize'] = [15, 10]

# +
# plot results
"""
plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks)

plot_initial_momenta_and_landmarks(final_momenta.flatten(), registered_landmarks.flatten(),
                                   min_x=0., max_x=7., min_y=-2., max_y=4.)

time_evolution_momenta = result['time_evolution_momenta']
time_evolution_positions = result['time_evolution_positions']
plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
                           min_x=0., max_x=7., min_y=-2., max_y=4.)

ani = animate_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
                                    min_x=0., max_x=7., min_y=-2., max_y=4.)
"""

nx = 10#70
ny = 10#60
mins = np.array([0., -2.])
maxs = np.array([7., 5.])
spatial_shape = (nx, ny)
flow = compute_time_evolution_of_diffeomorphisms(final_momenta, input_landmarks,
                                                 mins=mins, maxs=maxs, spatial_shape=spatial_shape)
flow.plot("Flow")

def set_landmarks_in_image(img, landmarks, sigma=1.):
    x, y = np.meshgrid(np.linspace(mins[0], maxs[0], nx), np.linspace(mins[1], maxs[1], ny))
    for i, l in enumerate(landmarks):
        dst = (x - l[0])**2 + (y - l[1])**2
        img += (i + 1.) * np.exp(-(dst / (2.0 * sigma**2)))

sigma = 0.5
image = ScalarFunction(spatial_shape)
target_image = ScalarFunction(spatial_shape)
set_landmarks_in_image(image, input_landmarks, sigma=sigma)
set_landmarks_in_image(target_image, target_landmarks, sigma=sigma)

image.plot("Original image")
target_image.plot("Target image")
resulting_image = push_forward(image, flow)
resulting_image.plot("Transformed image")
plt.show()

print(f"Input: {input_landmarks}")
print(f"Target: {target_landmarks}")
print(f"Result: {registered_landmarks}")
rel_error = (np.linalg.norm(target_landmarks - registered_landmarks)
             / np.linalg.norm(target_landmarks))
print(f"Relative norm of difference: {rel_error}")

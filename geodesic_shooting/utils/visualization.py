import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from geodesic_shooting.utils.kernels import GaussianKernel
from geodesic_shooting.landmark_shooting import construct_vector_field


def plot_initial_momenta_and_landmarks(momenta, positions, kernel=GaussianKernel(),
                                       N=30, title='', axis=None, landmark_size=50, arrow_scale=2.):
    """Plot the given initial momenta and landmarks.

    Parameters
    ----------
    momenta
        Array containing the landmark momenta.
    positions
        Array containing the landmark positions.
    kernel
        Kernel to use for extending the vector field to the whole domain.
    N
        Size of the vector field grid.
    title
        Title of the plot.
    axis
        If not `None`, the function is plotted on the provided axis.
    landmark_size
        Size of the landmarks.
    arrow_scale
        Scaling factor for the arrows.

    Returns
    -------
    The created plot.
    """
    assert momenta.ndim == 2
    assert momenta.shape == positions.shape
    assert momenta.shape[1] == 2, "Only implemented for 2d!"

    created_figure = False
    if not axis:
        created_figure = True
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')
    axis.set_title(title)

    vector_field = construct_vector_field(momenta, positions, kernel=kernel)

    xs = np.array([[x for x in np.linspace(0., 1., N)] for _ in np.linspace(0., 1., N)])
    ys = np.array([[y for _ in np.linspace(0., 1., N)] for y in np.linspace(0., 1., N)])
    vector_field_x = np.array([[vector_field(np.array([x, y]))[0]
                               for x in np.linspace(0., 1., N)]
                              for y in np.linspace(0., 1., N)])
    vector_field_y = np.array([[vector_field(np.array([x, y]))[1]
                               for x in np.linspace(0., 1., N)]
                              for y in np.linspace(0., 1., N)])

    axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

    colors = [f'C{i}' for i in range(len(positions))]

    axis.scatter(positions[:, 0], positions[:, 1], s=landmark_size, color=colors)
    axis.quiver(positions[:, 0], positions[:, 1], momenta[:, 0], momenta[:, 1],
                color=colors, scale=arrow_scale, angles='xy', scale_units='xy')

    if created_figure:
        return fig, axis
    return axis


def plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=GaussianKernel(),
                               N=30, title='', axis=None, landmark_size=10, arrow_scale=2.,
                               show_vector_fields=True):
    """Plot the trajectories of the landmarks.

    Parameters
    ----------
    time_evolution_momenta
        Array containing the landmark momenta at different time instances.
    time_evolution_positions
        Array containing the landmark positions at different time instances.
    kernel
        Kernel to use for extending the vector field to the whole domain.
    N
        Size of the vector field grid.
    title
        Title of the plot.
    axis
        If not `None`, the function is plotted on the provided axis.
    landmark_size
        Size of the landmarks.
    arrow_scale
        Scaling factor for the arrows.
    show_vector_fields
        Determines whether to also show the initial vector field.

    Returns
    -------
    The created plot.
    """
    assert time_evolution_momenta.ndim == 3
    assert time_evolution_momenta.shape == time_evolution_positions.shape

    dim = 2
    assert time_evolution_momenta.shape[-1] == dim

    created_figure = False
    if not axis:
        created_figure = True
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')
    axis.set_title(title)

    if show_vector_fields:
        vector_field = construct_vector_field(time_evolution_momenta[0],
                                              time_evolution_positions[0],
                                              kernel=kernel)

        xs = np.array([[x for x in np.linspace(0., 1., N)] for _ in np.linspace(0., 1., N)])
        ys = np.array([[y for _ in np.linspace(0., 1., N)] for y in np.linspace(0., 1., N)])
        vector_field_x = np.array([[vector_field(np.array([x, y]))[0] for x in np.linspace(0., 1., N)]
                                  for y in np.linspace(0., 1., N)])
        vector_field_y = np.array([[vector_field(np.array([x, y]))[1] for x in np.linspace(0., 1., N)]
                                  for y in np.linspace(0., 1., N)])

        axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

    colors = ([f'C{i}' for i in range(len(time_evolution_positions[0]))]
              * len(time_evolution_positions))

    axis.scatter(time_evolution_positions.reshape((-1, dim))[:, 0], time_evolution_positions.reshape((-1, dim))[:, 1],
                 s=landmark_size, color=colors)

    if created_figure:
        return fig, axis
    return axis


def animate_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=GaussianKernel(),
                                  N=30, title='', landmark_size=10, arrow_scale=2.):
    """Animate the trajectories of the landmarks.

    Parameters
    ----------
    time_evolution_momenta
        Array containing the landmark momenta at different time instances.
    time_evolution_positions
        Array containing the landmark positions at different time instances.
    kernel
        Kernel to use for extending the vector field to the whole domain.
    N
        Size of the vector field grid.
    title
        Title of the plot.
    landmark_size
        Size of the landmarks.
    arrow_scale
        Scaling factor for the arrows.

    Returns
    -------
    The created plot.
    """
    assert time_evolution_momenta.ndim == 3
    assert time_evolution_momenta.shape == time_evolution_positions.shape

    dim = 2
    assert time_evolution_momenta.shape[-1] == dim

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')
    axis.set_title(title)

    def plot_positions_and_velocity_field(momenta, positions):
        vector_field = construct_vector_field(momenta[-1], positions[-1], kernel=kernel)

        xs = np.array([[x for x in np.linspace(0., 1., N)]
                       for _ in np.linspace(0., 1., N)])
        ys = np.array([[y for _ in np.linspace(0., 1., N)]
                       for y in np.linspace(0., 1., N)])
        vector_field_x = np.array([[vector_field(np.array([x, y]))[0]
                                   for x in np.linspace(0., 1., N)]
                                  for y in np.linspace(0., 1., N)])
        vector_field_y = np.array([[vector_field(np.array([x, y]))[1]
                                   for x in np.linspace(0., 1., N)]
                                  for y in np.linspace(0., 1., N)])

        axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

        colors = ([f'C{i}' for i in range(len(positions[0].reshape((-1, dim))))]
                  * len(positions))

        axis.scatter(positions.reshape((-1, dim))[:, 0], positions.reshape((-1, dim))[:, 1],
                     s=landmark_size, color=colors)

        return fig

    def animate(i):
        axis.clear()
        pos = time_evolution_positions[:i+1]
        mom = time_evolution_momenta[:i+1]
        plot_positions_and_velocity_field(mom, pos)

    ani = animation.FuncAnimation(fig, animate, frames=time_evolution_positions.shape[0], interval=100)
    return ani


def plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks,
                            title='', axis=None, landmark_size=50):
    """Plot the results of the matching of landmarks.

    Parameters
    ----------
    input_landmarks
        Positions of the input landmarks.
    target_landmarks
        Positions of the target landmarks.
    registered_landmarks
        Positions of the registered landmarks.
    title
        Title of the plot.
    axis
        If not `None`, the function is plotted on the provided axis.
    landmark_size
        Size of the landmarks.

    Returns
    -------
    The created plot.
    """
    assert input_landmarks.ndim == 2
    assert input_landmarks.shape == target_landmarks.shape == registered_landmarks.shape

    created_figure = False
    if not axis:
        created_figure = True
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')
    axis.set_title(title)

    colors = [f'C{i}' for i in range(len(input_landmarks))]

    axis.scatter(input_landmarks[:, 0], input_landmarks[:, 1],
                 s=landmark_size, color=colors, marker='o', label="Input landmark")
    axis.scatter(target_landmarks[:, 0], target_landmarks[:, 1],
                 s=landmark_size, color=colors, marker='*', label="Target landmark")
    axis.scatter(registered_landmarks[:, 0], registered_landmarks[:, 1],
                 s=landmark_size, color=colors, marker='s', label="Registered landmark")

    plt.legend()

    if created_figure:
        return fig, axis
    return axis


def plot_landmarks(input_landmarks, target_landmarks, registered_landmarks,
                   title='', axis=None, landmark_size=50):
    """Plot the results of the matching of landmarks.

    Parameters
    ----------
    input_landmarks
        Positions of the input landmarks.
    target_landmarks
        Positions of the target landmarks.
    registered_landmarks
        Positions of the registered landmarks.
    title
        Title of the plot.
    axis
        If not `None`, the function is plotted on the provided axis.
    landmark_size
        Size of the landmarks.

    Returns
    -------
    The created plot.
    """
    assert input_landmarks.ndim == 2
    assert input_landmarks.shape == registered_landmarks.shape
    assert target_landmarks.ndim == 2

    created_figure = False
    if not axis:
        created_figure = True
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')
    axis.set_title(title)

    colors = [f'C{i}' for i in range(len(input_landmarks))]

    axis.scatter(input_landmarks[:, 0], input_landmarks[:, 1],
                 s=landmark_size, color=colors, marker='o', label="Input landmark")
    axis.scatter(target_landmarks[:, 0], target_landmarks[:, 1],
                 s=landmark_size, marker='*', label="Target landmark")
    axis.scatter(registered_landmarks[:, 0], registered_landmarks[:, 1],
                 s=landmark_size, color=colors, marker='s', label="Registered landmark")

    plt.legend()

    if created_figure:
        return fig, axis
    return axis

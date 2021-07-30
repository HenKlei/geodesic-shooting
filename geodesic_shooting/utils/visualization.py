import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from geodesic_shooting.utils.kernels import GaussianKernel


def plot_warpgrid(warp, title='', interval=2, show_axis=False):
    """Plot the given warpgrid.

    Parameters
    ----------
    warp
        Warp grid to plot.
    title
        Title of the plot.
    interval
        Interval in which to sample.
    show_axis
        Determines whether or not to show the axes.

    Returns
    -------
    The created plot.
    """
    assert warp.shape[0] == 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    if show_axis is False:
        axis.set_axis_off()

    axis.invert_yaxis()
    axis.set_aspect('equal')
    axis.set_title(title)

    for row in range(0, warp.shape[1], interval):
        axis.plot(warp[1, row, :], warp[0, row, :], 'k')
    for col in range(0, warp.shape[2], interval):
        axis.plot(warp[1, :, col], warp[0, :, col], 'k')

    return fig


def plot_vector_field(vector_field, title='', interval=1, show_axis=False):
    """Plot the given (two-dimensional) vector field.

    Parameters
    ----------
    vector_field
        Field to plot.
    title
        Title of the plot.
    interval
        Interval in which to sample.
    show_axis
        Determines whether or not to show the axes.


    Returns
    -------
    The created plot.
    """
    assert vector_field.shape[0] == 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    if show_axis is False:
        axis.set_axis_off()

    axis.set_aspect('equal')
    axis.set_title(title)

    axis.quiver(-vector_field[0, ::interval, ::interval], vector_field[1, ::interval, ::interval])

    return fig


def construct_vector_field(momenta, positions, kernel=GaussianKernel()):
    """Computes the vector field corresponding to the given positions and momenta.

    Parameters
    ----------
    momenta
        Array containing the momenta of the landmarks.
    positions
        Array containing the positions of the landmarks.

    Returns
    -------
    Function that can be evaluated at any point of the space.
    """
    assert positions.ndim == 2
    assert positions.shape == momenta.shape

    def vector_field(x):
        result = np.zeros(positions.shape[1])
        for q, p in zip(positions, momenta):
            result += kernel(x, q) @ p
        return result

    return vector_field


def plot_initial_momenta_and_landmarks(momenta, landmarks, kernel=GaussianKernel(),
                                       min_x=-1., max_x=1., min_y=-1., max_y=1., N=30,
                                       landmark_size=50, arrow_scale=2.):
    assert momenta.ndim == 1
    assert momenta.shape == landmarks.shape

    dim = 2

    landmarks = landmarks.reshape((-1, dim))
    momenta = momenta.reshape((-1, dim))

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    vector_field = construct_vector_field(momenta, landmarks)

    xs = np.array([[x for x in np.linspace(min_x, max_x, N)] for _ in np.linspace(min_y, max_y, N)])
    ys = np.array([[y for _ in np.linspace(min_x, max_x, N)] for y in np.linspace(min_y, max_y, N)])
    vector_field_x = np.array([[vector_field(np.array([x, y]))[0]
                               for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])
    vector_field_y = np.array([[vector_field(np.array([x, y]))[1]
                               for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])

    axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

    colors = [f'C{i}' for i in range(len(landmarks))]

    axis.scatter(landmarks[:, 0], landmarks[:, 1], s=landmark_size, color=colors)
    axis.quiver(landmarks[:, 0], landmarks[:, 1], momenta[:, 0], momenta[:, 1],
                color=colors, scale=arrow_scale, angles='xy', scale_units='xy')

    return fig


def plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=GaussianKernel(),
                               min_x=-1., max_x=1., min_y=-1., max_y=1., N=30,
                               landmark_size=10, arrow_scale=2.):
    assert time_evolution_momenta.ndim == 2
    assert time_evolution_momenta.shape == time_evolution_positions.shape

    dim = 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    vector_field = construct_vector_field(time_evolution_momenta[0].reshape((-1, dim)),
                                          time_evolution_positions[0].reshape((-1, dim)))

    xs = np.array([[x for x in np.linspace(min_x, max_x, N)] for _ in np.linspace(min_y, max_y, N)])
    ys = np.array([[y for _ in np.linspace(min_x, max_x, N)] for y in np.linspace(min_y, max_y, N)])
    vector_field_x = np.array([[vector_field(np.array([x, y]))[0] for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])
    vector_field_y = np.array([[vector_field(np.array([x, y]))[1] for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])

    axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

    colors = ([f'C{i}' for i in range(len(time_evolution_positions[0].reshape((-1, dim))))]
              * len(time_evolution_positions))

    axis.scatter(time_evolution_positions.reshape((-1, dim))[:, 0], time_evolution_positions.reshape((-1, dim))[:, 1],
                 s=landmark_size, color=colors)

    return fig


def animate_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=GaussianKernel(),
                                  min_x=-1., max_x=1., min_y=-1., max_y=1., N=30,
                                  landmark_size=10, arrow_scale=2.):
    assert time_evolution_momenta.ndim == 2
    assert time_evolution_momenta.shape == time_evolution_positions.shape

    dim = 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    def plot_positions_and_velocity_field(momenta, positions):
        vector_field = construct_vector_field(momenta[-1], positions[-1])

        xs = np.array([[x for x in np.linspace(min_x, max_x, N)]
                       for _ in np.linspace(min_y, max_y, N)])
        ys = np.array([[y for _ in np.linspace(min_x, max_x, N)]
                       for y in np.linspace(min_y, max_y, N)])
        vector_field_x = np.array([[vector_field(np.array([x, y]))[0]
                                   for x in np.linspace(min_x, max_x, N)]
                                  for y in np.linspace(min_y, max_y, N)])
        vector_field_y = np.array([[vector_field(np.array([x, y]))[1]
                                   for x in np.linspace(min_x, max_x, N)]
                                  for y in np.linspace(min_y, max_y, N)])

        axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

        colors = ([f'C{i}' for i in range(len(positions[0].reshape((-1, dim))))]
                  * len(positions))

        axis.scatter(positions.reshape((-1, dim))[:, 0], positions.reshape((-1, dim))[:, 1],
                     s=landmark_size, color=colors)

        return fig

    def animate(i):
        axis.clear()
        pos = time_evolution_positions[:i+1].reshape((i+1, -1, dim))
        mom = time_evolution_momenta[:i+1].reshape((i+1, -1, dim))
        plot_positions_and_velocity_field(mom, pos)

    ani = animation.FuncAnimation(fig, animate, frames=time_evolution_positions.shape[0], interval=100)
    return ani


def plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks, landmark_size=50):
    assert input_landmarks.ndim == 2
    assert input_landmarks.shape == target_landmarks.shape == registered_landmarks.shape

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    colors = [f'C{i}' for i in range(len(input_landmarks))]

    axis.scatter(input_landmarks[:, 0], input_landmarks[:, 1],
                 s=landmark_size, color=colors, marker='o', label="Input landmark")
    axis.scatter(target_landmarks[:, 0], target_landmarks[:, 1],
                 s=landmark_size, color=colors, marker='*', label="Target landmark")
    axis.scatter(registered_landmarks[:, 0], registered_landmarks[:, 1],
                 s=landmark_size, color=colors, marker='s', label="Registered landmark")

    plt.legend()

    return fig

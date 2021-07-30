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


def plot_initial_momenta_and_landmarks(momenta, landmarks, kernel=GaussianKernel(), dim=2,
                                       min_x=-1., max_x=1., min_y=-1., max_y=1., N=30,
                                       landmark_size=50, arrow_scale=2.):
    assert momenta.ndim == 1
    assert momenta.shape == landmarks.shape

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    vector_field = construct_vector_field(momenta.reshape((-1, dim)), landmarks.reshape((-1, dim)))

    xs = np.array([[x for x in np.linspace(min_x, max_x, N)] for _ in np.linspace(min_y, max_y, N)])
    ys = np.array([[y for _ in np.linspace(min_x, max_x, N)] for y in np.linspace(min_y, max_y, N)])
    vector_field_x = np.array([[vector_field(np.array([x, y]))[0]
                               for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])
    vector_field_y = np.array([[vector_field(np.array([x, y]))[1]
                               for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])

    axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

    for i, (landmark, momentum) in enumerate(zip(landmarks.reshape((-1, dim)), momenta.reshape((-1, dim)))):
        axis.scatter(landmark[0], landmark[1], s=landmark_size, color=f'C{i}')
        axis.arrow(landmark[0], landmark[1], momentum[0], momentum[1],
                   head_width=0.05, color=f'C{i}')

    return fig


def plot_landmark_trajectories(momenta, positions, kernel=GaussianKernel(), dim=2,
                               min_x=-1., max_x=1., min_y=-1., max_y=1., N=30,
                               landmark_size=10, arrow_scale=2.):
    assert positions.ndim == 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    vector_field = construct_vector_field(momenta[0].reshape((-1, dim)), positions[0].reshape((-1, dim)))

    xs = np.array([[x for x in np.linspace(-2., 2., N)] for _ in np.linspace(-2., 2., N)])
    ys = np.array([[y for _ in np.linspace(-2., 2., N)] for y in np.linspace(-2., 2., N)])
    vector_field_x = np.array([[vector_field(np.array([x, y]))[0] for x in np.linspace(-2., 2., N)]
                              for y in np.linspace(-2., 2., N)])
    vector_field_y = np.array([[vector_field(np.array([x, y]))[1] for x in np.linspace(-2., 2., N)]
                              for y in np.linspace(-2., 2., N)])

    axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

    for pos in positions:
        pos = pos.reshape((-1, dim))
        for i, landmark in enumerate(pos):
            axis.scatter(landmark[0], landmark[1], s=landmark_size, color=f'C{i}')

    return fig


def animate_landmark_trajectories(time_dependent_momenta, time_dependent_positions, kernel=GaussianKernel(), dim=2,
                                  min_x=-1., max_x=1., min_y=-1., max_y=1., N=30,
                                  max_landmark_size=50, min_landmark_size=10, arrow_scale=2.):
    assert time_dependent_momenta.ndim == 2
    assert time_dependent_momenta.shape == time_dependent_positions.shape

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

        for j, pos in enumerate(positions):
            for i, landmark in enumerate(pos):
                size = max_landmark_size if j == len(positions) - 1 else min_landmark_size
                axis.scatter(landmark[0], landmark[1],
                             s=size, color=f'C{i}')

        return fig

    def animate(i):
        axis.clear()
        pos = time_dependent_positions[:i+1].reshape((i+1, -1, dim))
        mom = time_dependent_momenta[:i+1].reshape((i+1, -1, dim))
        plot_positions_and_velocity_field(mom, pos)

    ani = animation.FuncAnimation(fig, animate, frames=time_dependent_positions.shape[0], interval=1000)
    return ani

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from geodesic_shooting.utils.kernels import GaussianKernel


def plot_warpgrid(warp, title='', interval=2, show_axis=False, tight_layout=True, invert_yaxis=True):
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
    assert warp.ndim == 3
    assert warp.shape[0] == 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    if show_axis is False:
        axis.set_axis_off()

    if invert_yaxis:
        axis.invert_yaxis()
    axis.set_aspect('equal')
    axis.set_title(title)

    for row in range(0, warp.shape[1], interval):
        axis.plot(warp[0, row, :], warp[1, row, :], 'k')
    for col in range(0, warp.shape[2], interval):
        axis.plot(warp[0, :, col], warp[1, :, col], 'k')

    if tight_layout:
        fig.tight_layout()

    return fig


def plot_vector_field(vector_field, title='', interval=1, show_axis=False, tight_layout=True):
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
    axis.invert_yaxis()

    axis.quiver(vector_field[0, ::interval, ::interval].T, vector_field[1, ::interval, ::interval].T)

    if tight_layout:
        fig.tight_layout()

    return fig


def plot_image(image, title='', three_d=False, tight_layout=True):
    assert image.ndim == 2

    fig = plt.figure()
    if three_d:
        axis = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        axis = fig.add_subplot(1, 1, 1)

    if not three_d:
        axis.set_aspect('equal')
    axis.set_title(title)

    shape = image.shape
    grid = np.mgrid[0:shape[0], 0:shape[1]]

    min_val = np.min(image)
    max_val = np.max(image)

    for i, z_val in enumerate(image):
        x_val = grid[0][i]
        y_val = grid[1][i]
        if three_d:
            scatter = axis.scatter(x_val, y_val, z_val, c=z_val, vmin=min_val, vmax=max_val)
        else:
            scatter = axis.scatter(x_val, y_val, c=z_val, vmin=min_val, vmax=max_val)

    fig.colorbar(scatter)

    if tight_layout:
        fig.tight_layout()

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


def plot_initial_momenta_and_landmarks(momenta, positions, kernel=GaussianKernel(),
                                       min_x=-1., max_x=1., min_y=-1., max_y=1., N=30,
                                       title='', landmark_size=50, arrow_scale=2.):
    """Plot the given initial momenta and landmarks.

    Parameters
    ----------
    momenta
        Array containing the landmark momenta.
    positions
        Array containing the landmark positions.
    kernel
        Kernel to use for extending the vector field to the whole domain.
    min_x
        Minimum x-value.
    max_x
        Maximum x-value.
    min_y
        Minimum y-value.
    max_y
        Maximum y-value.
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
    assert momenta.ndim == 1
    assert momenta.shape == positions.shape

    dim = 2

    positions = positions.reshape((-1, dim))
    momenta = momenta.reshape((-1, dim))

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')
    axis.set_title(title)

    vector_field = construct_vector_field(momenta, positions)

    xs = np.array([[x for x in np.linspace(min_x, max_x, N)] for _ in np.linspace(min_y, max_y, N)])
    ys = np.array([[y for _ in np.linspace(min_x, max_x, N)] for y in np.linspace(min_y, max_y, N)])
    vector_field_x = np.array([[vector_field(np.array([x, y]))[0]
                               for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])
    vector_field_y = np.array([[vector_field(np.array([x, y]))[1]
                               for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])

    axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=arrow_scale, angles='xy', scale_units='xy')

    colors = [f'C{i}' for i in range(len(positions))]

    axis.scatter(positions[:, 0], positions[:, 1], s=landmark_size, color=colors)
    axis.quiver(positions[:, 0], positions[:, 1], momenta[:, 0], momenta[:, 1],
                color=colors, scale=arrow_scale, angles='xy', scale_units='xy')

    return fig


def plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions, kernel=GaussianKernel(),
                               min_x=-1., max_x=1., min_y=-1., max_y=1., N=30,
                               title='', landmark_size=10, arrow_scale=2.):
    """Plot the trajectories of the landmarks.

    Parameters
    ----------
    time_evolution_momenta
        Array containing the landmark momenta at different time instances.
    time_evolution_positions
        Array containing the landmark positions at different time instances.
    kernel
        Kernel to use for extending the vector field to the whole domain.
    min_x
        Minimum x-value.
    max_x
        Maximum x-value.
    min_y
        Minimum y-value.
    max_y
        Maximum y-value.
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
    assert time_evolution_momenta.ndim == 2
    assert time_evolution_momenta.shape == time_evolution_positions.shape

    dim = 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')
    axis.set_title(title)

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
                                  title='', landmark_size=10, arrow_scale=2.):
    """Animate the trajectories of the landmarks.

    Parameters
    ----------
    time_evolution_momenta
        Array containing the landmark momenta at different time instances.
    time_evolution_positions
        Array containing the landmark positions at different time instances.
    kernel
        Kernel to use for extending the vector field to the whole domain.
    min_x
        Minimum x-value.
    max_x
        Maximum x-value.
    min_y
        Minimum y-value.
    max_y
        Maximum y-value.
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
    assert time_evolution_momenta.ndim == 2
    assert time_evolution_momenta.shape == time_evolution_positions.shape

    dim = 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')
    axis.set_title(title)

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


def plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks, title='', landmark_size=50):
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
    landmark_size
        Size of the landmarks.

    Returns
    -------
    The created plot.
    """
    assert input_landmarks.ndim == 2
    assert input_landmarks.shape == target_landmarks.shape == registered_landmarks.shape

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

    return fig


def animate_warpgrids(time_evolution_warp, min_x=-1., max_x=1., min_y=-1., max_y=1.,
                      title='', interval=1, show_axis=True):
    """Animate the trajectories of the landmarks.

    Parameters
    ----------
    time_evolution_momenta
        Array containing the landmark momenta at different time instances.
    time_evolution_positions
        Array containing the landmark positions at different time instances.
    kernel
        Kernel to use for extending the vector field to the whole domain.
    min_x
        Minimum x-value.
    max_x
        Maximum x-value.
    min_y
        Minimum y-value.
    max_y
        Maximum y-value.
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
    assert time_evolution_warp.ndim == 4
    assert time_evolution_warp.shape[1] == 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    if show_axis is False:
        axis.set_axis_off()

    axis.set_aspect('equal')
    axis.set_title(title)

    axis.set_xlim([min_x, max_x])
    axis.set_ylim([min_y, max_y])

    def plot_warp(warp):
        assert warp.ndim == 3
        assert warp.shape[0] == 2
        for row in range(0, warp.shape[1], interval):
            axis.plot(warp[0, row, :], warp[1, row, :], 'k')
        for col in range(0, warp.shape[2], interval):
            axis.plot(warp[0, :, col], warp[1, :, col], 'k')
        return fig

    def animate(i):
        axis.clear()
        axis.set_xlim([min_x, max_x])
        axis.set_ylim([min_y, max_y])
        w = time_evolution_warp[i].reshape((-1, 2)).T.reshape(time_evolution_warp[i].shape)
        plot_warp(w)

    ani = animation.FuncAnimation(fig, animate, frames=time_evolution_warp.shape[0], interval=100)
    return ani

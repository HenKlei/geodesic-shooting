import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def plot_warpgrid(warp, title='', interval=2, show_axis=False, tight_layout=True):
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

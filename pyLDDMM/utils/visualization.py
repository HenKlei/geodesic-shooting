import matplotlib.pyplot as plt


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


def plot_vector_field(vector_field, title='', interval=1):
    """Plot the given (two-dimensional) vector field.

    Parameters
    ----------
    vector_field
        Field to plot.
    title
        Title of the plot.
    interval
        Interval in which to sample.

    Returns
    -------
    The created plot.
    """
    assert vector_field.shape[0] == 2

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_aspect('equal')
    axis.set_title(title)

    axis.quiver(vector_field[0, ::interval, ::interval], vector_field[1, ::interval, ::interval])

    return fig

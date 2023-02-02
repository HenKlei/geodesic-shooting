import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from geodesic_shooting.core.base import BaseFunction, BaseTimeDependentFunction
from geodesic_shooting.utils import sampler, grid


class Diffeomorphism(BaseFunction):
    """Class that represents a vector-valued function, i.e. a vector field."""
    def __init__(self, spatial_shape=(), data=None):
        """Constructor.

        Parameters
        ----------
        spatial_shape
            Spatial shape of the underlying domain, i.e. number of components
            in each coordinate direction.
        data
            numpy-array containing the values of the `VectorField`. If `None`,
            one has to provide `spatial_shape` and a numpy-array containing zeros
            with shape `(*spatial_shape, dim)` is created as data, where `dim` is
            the dimension of the underlying domain (given as `len(spatial_shape)`).
            If not `None`, either `spatial_shape` is the empty tuple or the first
            components of the shape of `data` (without the last component)
            fit to the provided `spatial_shape`.
        """
        super().__init__(spatial_shape, data)

        assert len(self.spatial_shape) == self._data.shape[-1]

    def _compute_spatial_shape(self, spatial_shape, data):
        if data is None:
            return spatial_shape
        return data.shape[0:-1]

    def _compute_full_shape(self):
        return (*self.spatial_shape, self.dim)

    def plot_as_warpgrid(self, title="", interval=1, show_axis=False, show_identity_grid=True, axis=None,
                         show_displacement_vectors=False, color_length=False):
        """Plots the `VectorField` as a warpgrid using `matplotlib`.

        Parameters
        ----------
        title
            The title of the plot.
        interval
            Interval in which to sample.
        show_axis
            Determines whether or not to show the axes.
        show_identity_grid
            Determines whether or not to show the underlying identity grid.
        axis
            If not `None`, the function is plotted on the provided axis.
        show_displacement_vectors
            Determines whether or not to show the corresponding displacement
            vectors.
        color_length
            Determines whether or not to show the lengths of the vectors using
            different colors.
            Only used if `show_displacement_vectors` is `True`.

        Returns
        -------
        If `axis` is None, the created figure is returned, otherwise the axis
        is returned.
        """
        assert self.dim == 2

        created_figure = False
        if not axis:
            created_figure = True
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1)

        def plot_grid(x, y, **kwargs):
            segs1 = np.stack([x, y], axis=-1)
            segs2 = segs1.transpose(1, 0, 2)
            axis.add_collection(LineCollection(segs1, **kwargs))
            axis.add_collection(LineCollection(segs2, **kwargs))
            axis.autoscale()

        identity_grid = grid.coordinate_grid(self.spatial_shape)
        grid_x, grid_y = identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1]
        grid_x = np.vstack([grid_x, identity_grid[-1, ::interval, 0][np.newaxis, ...]])
        grid_x = np.hstack([grid_x, np.hstack([identity_grid[::interval, -1, 0],
                                               identity_grid[-1, -1, 0]])[..., np.newaxis]])
        grid_y = np.vstack([grid_y, identity_grid[-1, ::interval, 1][np.newaxis, ...]])
        grid_y = np.hstack([grid_y, np.hstack([identity_grid[::interval, -1, 1],
                                               identity_grid[-1, -1, 1]])[..., np.newaxis]])
        if show_identity_grid:
            plot_grid(grid_x, grid_y, color="lightgrey")

        dist_x, dist_y = self[::interval, ::interval, 0], self[::interval, ::interval, 1]
        dist_x = np.vstack([dist_x, self[-1, ::interval, 0][np.newaxis, ...]])
        dist_x = np.hstack([dist_x, np.hstack([self[::interval, -1, 0], self[-1, -1, 0]])[..., np.newaxis]])
        dist_y = np.vstack([dist_y, self[-1, ::interval, 1][np.newaxis, ...]])
        dist_y = np.hstack([dist_y, np.hstack([self[::interval, -1, 1], self[-1, -1, 1]])[..., np.newaxis]])
        plot_grid(dist_x, dist_y, color="C0")

        if show_displacement_vectors:
            self.plot(scale=1., axis=axis, zorder=2, color_length=color_length)

        if show_axis is False:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if created_figure:
            return fig, axis
        return axis

    def save(self, filepath, title="", interval=1, show_axis=False, show_identity_grid=True,
             show_displacement_vectors=False, color_length=False):
        """Saves the plot of the `VectorField` produced by the `plot`-function.

        Parameters
        ----------
        filepath
            Path to save the plot to.
        title
            Title of the plot.
        interval
            Interval in which to sample.
        show_axis
            Determines whether or not to show the axes.
        show_identity_grid
            Determines whether or not to show the underlying identity grid.
        show_displacement_vectors
            Determines whether or not to show the corresponding displacement
            vectors.
        color_length
            Determines whether or not to show the lengths of the vectors using
            different colors.
            Only used if `show_displacement_vectors` is `True`.
        """
        try:
            fig, _ = self.plot_as_warpgrid(title=title, interval=interval, show_axis=show_axis,
                                           show_identity_grid=show_identity_grid, axis=None,
                                           show_displacement_vectors=show_displacement_vectors,
                                           color_length=color_length)
            fig.savefig(filepath)
            plt.close(fig)
        except Exception:
            pass


class TimeDependentDiffeomorphism(BaseTimeDependentFunction):
    """Class that represents a time-dependent vector field."""
    def __init__(self, spatial_shape=(), time_steps=1, data=None):
        """Constructor.

        Parameters
        ----------
        spatial_shape
            Spatial shape of the underlying domain, i.e. number of components
            in each coordinate direction.
        time_steps
            Number of time steps represented in this vector field.
        data
            numpy-array containing the values of the `TimeDependentVectorFieldVectorField`.
            If `None`, one has to provide `spatial_shape` and `time_steps`, and a
            list containing for each time step a `VectorField` of zeros with shape
            `(*spatial_shape, dim)` is created as data, where `dim` is the dimension of
            the underlying domain (given as `len(spatial_shape)`).
            If not `None`, either a numpy-array or a `TimeDependentVectorField` or a list
            of `VectorField`s has to be provided.
        """
        super().__init__(spatial_shape, time_steps, data)

    def plot(self, title="", interval=1, frequency=1, show_axis=False, show_identity_grid=True, axis=None):
        assert self.dim == 2

        created_figure = False
        if not axis:
            created_figure = True
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1)

        def plot_grid(x, y, **kwargs):
            segs1 = np.stack([x, y], axis=-1)
            segs2 = segs1.transpose(1, 0, 2)
            axis.add_collection(LineCollection(segs1, **kwargs))
            axis.add_collection(LineCollection(segs2, **kwargs))
            axis.autoscale()

        identity_grid = grid.coordinate_grid(self.spatial_shape)
        grid_x, grid_y = identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1]
        if show_identity_grid:
            plot_grid(grid_x, grid_y, color="lightgrey", zorder=1)

        plot_grid(self[-1, ::interval, ::interval, 0], self[-1, ::interval, ::interval, 1], color="C0", zorder=3)

        for t in range(0, self.time_steps, frequency):
            axis.scatter(self[t, ::interval, ::interval, 0], self[t, ::interval, ::interval, 1], c="C1", zorder=2)

        if show_axis is False:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if created_figure:
            return fig, axis
        return axis

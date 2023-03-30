import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

from geodesic_shooting.core.base import BaseFunction, BaseTimeDependentFunction
from geodesic_shooting.core.functions import ScalarFunction
from geodesic_shooting.utils import grid


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

    def set_inverse(self, diffeomorphism):
        """Assigns a variable with the inverse of the diffeomorphism.

        The inverse is usually computed by integrating the corresponding
        vector field backward in time.

        Parameters
        ----------
        diffeomorphism
            The inverse diffeomorphism.
        """
        assert isinstance(diffeomorphism, Diffeomorphism)
        assert diffeomorphism.spatial_shape == self.spatial_shape
        self.inverse_diffeomorphism = diffeomorphism

    @property
    def inverse(self):
        """Returns the inverse of the diffeomorphism if available.

        Returns
        -------
        The inverse diffeomorphism.
        """
        assert hasattr(self, 'inverse_diffeomorphism'), "Inverse diffeomorphism not set."
        return self.inverse_diffeomorphism

    def plot(self, title="", interval=1, show_axis=True, show_identity_grid=True, axis=None,
             figsize=(10, 10), show_displacement_vectors=False, color_length=False):
        """Plots the `Diffeomorphism` as a warpgrid using `matplotlib`.

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
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.
        show_displacement_vectors
            Determines whether or not to show the corresponding displacement
            vectors.
        color_length
            Determines whether or not to show the lengths of the vectors using
            different colors.
            Only used if `show_displacement_vectors` is `True`.

        Returns
        -------
        If `axis` is None, the created figure and the axis are returned,
        otherwise only the altered axis is returned.
        """
        assert self.dim == 2

        created_figure = False
        if not axis:
            created_figure = True
            fig, axis = plt.subplots(1, 1, figsize=figsize)

        def plot_grid(x, y, **kwargs):
            segs1 = np.stack([x, y], axis=-1)
            segs2 = segs1.transpose(1, 0, 2)
            axis.add_collection(LineCollection(segs1, **kwargs))
            axis.add_collection(LineCollection(segs2, **kwargs))
            axis.autoscale()

        identity_grid = grid.coordinate_grid(self.spatial_shape)
        identity_grid = np.stack([identity_grid[..., 0] / self.spatial_shape[0],
                                  identity_grid[..., 1] / self.spatial_shape[1]], axis=-1)
        grid_x, grid_y = identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1]
        grid_x = np.vstack([grid_x, identity_grid[-1, ::interval, 0][np.newaxis, ...]])
        grid_x = np.hstack([grid_x, np.hstack([identity_grid[::interval, -1, 0],
                                               identity_grid[-1, -1, 0]])[..., np.newaxis]])
        grid_y = np.vstack([grid_y, identity_grid[-1, ::interval, 1][np.newaxis, ...]])
        grid_y = np.hstack([grid_y, np.hstack([identity_grid[::interval, -1, 1],
                                               identity_grid[-1, -1, 1]])[..., np.newaxis]])
        if show_identity_grid:
            plot_grid(grid_x, grid_y, color="lightgrey")

        dist_x = self[::interval, ::interval, 0] / self.spatial_shape[0]
        dist_y = self[::interval, ::interval, 1] / self.spatial_shape[1]
        dist_x = np.vstack([dist_x, self[-1, ::interval, 0][np.newaxis, ...] / self.spatial_shape[0]])
        dist_x = np.hstack([dist_x, np.hstack([self[::interval, -1, 0],
                                               self[-1, -1, 0]])[..., np.newaxis] / self.spatial_shape[0]])
        dist_y = np.vstack([dist_y, self[-1, ::interval, 1][np.newaxis, ...] / self.spatial_shape[1]])
        dist_y = np.hstack([dist_y, np.hstack([self[::interval, -1, 1],
                                               self[-1, -1, 1]])[..., np.newaxis] / self.spatial_shape[1]])
        dist_x = grid_x + (dist_x - grid_x) * self.spatial_shape[0]
        dist_y = grid_y + (dist_y - grid_y) * self.spatial_shape[1]
        plot_grid(dist_x, dist_y, color="C0")

        if show_displacement_vectors:
            if color_length:
                colors = np.linalg.norm(self.to_numpy(), axis=-1)
                axis.quiver(identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1],
                            self[::interval, ::interval, 0]
                            - identity_grid[::interval, ::interval, 0] * self.spatial_shape[0],
                            self[::interval, ::interval, 1]
                            - identity_grid[::interval, ::interval, 1] * self.spatial_shape[1],
                            colors, scale_units='xy', units='xy', angles='xy', scale=1, zorder=2)
            else:
                axis.quiver(identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1],
                            self[::interval, ::interval, 0]
                            - identity_grid[::interval, ::interval, 0] * self.spatial_shape[0],
                            self[::interval, ::interval, 1]
                            - identity_grid[::interval, ::interval, 1] * self.spatial_shape[1],
                            scale_units='xy', units='xy', angles='xy', scale=1, zorder=2)

        if not show_axis:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if created_figure:
            return fig, axis
        return axis

    def save(self, filepath, title="", interval=1, show_axis=True, show_identity_grid=True,
             show_displacement_vectors=False, color_length=False, dpi=100):
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
        dpi
            The resolution in dots per inch.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
            for more details.
        """
        try:
            fig, _ = self.plot(title=title, interval=interval, show_axis=show_axis,
                               show_identity_grid=show_identity_grid, axis=None,
                               show_displacement_vectors=show_displacement_vectors,
                               color_length=color_length)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
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

    def plot(self, title="", interval=1, frequency=1, show_axis=True, show_identity_grid=True, axis=None,
             figsize=(10, 10)):
        """Plots the `TimeDependentDiffeomorphism` as trajectories of points using `matplotlib`.

        Parameters
        ----------
        title
            The title of the plot.
        interval
            Interval in which to sample.
        frequency
            Frequency in which to sample the points in the trajectories.
        show_axis
            Determines whether or not to show the axes.
        show_identity_grid
            Determines whether or not to show the underlying identity grid.
        axis
            If not `None`, the function is plotted on the provided axis.
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.

        Returns
        -------
        If `axis` is None, the created figure and the axis are returned,
        otherwise only the altered axis is returned.
        """
        assert self.dim == 2

        created_figure = False
        if not axis:
            created_figure = True
            fig, axis = plt.subplots(1, 1, figsize=figsize)

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
            plot_grid(grid_x, grid_y, color="lightgrey", zorder=1)

        def get_scaled_and_extended_grid_points(t):
            dist_x, dist_y = self[t, ::interval, ::interval, 0], self[t, ::interval, ::interval, 1]
            dist_x = np.vstack([dist_x, self[t, -1, ::interval, 0][np.newaxis, ...]])
            dist_x = np.hstack([dist_x, np.hstack([self[t, ::interval, -1, 0], self[t, -1, -1, 0]])[..., np.newaxis]])
            dist_y = np.vstack([dist_y, self[t, -1, ::interval, 1][np.newaxis, ...]])
            dist_y = np.hstack([dist_y, np.hstack([self[t, ::interval, -1, 1], self[t, -1, -1, 1]])[..., np.newaxis]])
            dist_x = grid_x + (dist_x - grid_x) * self.spatial_shape[0]
            dist_y = grid_y + (dist_y - grid_y) * self.spatial_shape[1]
            return dist_x, dist_y

        dist_x, dist_y = get_scaled_and_extended_grid_points(-1)
        plot_grid(dist_x, dist_y, color="C0", zorder=3)

        for t in range(0, self.time_steps, frequency):
            dist_x, dist_y = get_scaled_and_extended_grid_points(t)
            axis.scatter(dist_x, dist_y, c="C1", zorder=2)

        if not show_axis:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if created_figure:
            return fig, axis
        return axis

    def animate(self, title="", interval=1, show_axis=True, figsize=(10, 10)):
        """Animates the `TimeDependentDiffeomorphism`.

        Parameters
        ----------
        title
            The title of the plot.
        interval
            Interval in which to sample.
        show_axis
            Determines whether or not to show the axes.
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.

        Returns
        -------
        The animation object.
        """
        fig, axis = plt.subplots(1, 1, figsize=figsize)

        if not show_axis:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        def update(i):
            axis.clear()
            self[i].plot(title=title, interval=interval, axis=axis, show_axis=show_axis,
                         show_displacement_vectors=False)

        time_steps = self.time_steps

        class PauseAnimation:
            def __init__(self):
                self.ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=100)
                self.paused = False

                fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

            def toggle_pause(self, *args, **kwargs):
                if self.paused:
                    self.ani.resume()
                else:
                    self.ani.pause()
                self.paused = not self.paused

            def save(self, filename='animation.gif', writer='imagemagick', fps=10):
                self.ani.save(filename, writer=writer, fps=fps)

        ani = PauseAnimation()
        return ani

    def animate_transformation(self, function, title="", interval=1, show_axis=True, figsize=(10, 10),
                               show_restriction_boundary=True, restriction=np.s_[...]):
        """Animates the `TimeDependentDiffeomorphism` together with a transformed `ScalarFunction`.

        Parameters
        ----------
        function
            `ScalarFunction` to transform and plot during the animation.
        title
            The title of the plot.
        interval
            Interval in which to sample.
        show_axis
            Determines whether or not to show the axes.
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.

        Returns
        -------
        The animation object.
        """
        assert isinstance(function, ScalarFunction)

        fig, axis = plt.subplots(1, 1, figsize=figsize)

        if not show_axis:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        def update(i):
            axis.clear()
            function.push_forward(self[i]).plot(axis=axis, show_axis=show_axis,
                                                show_restriction_boundary=show_restriction_boundary,
                                                restriction=restriction)
            self[i].plot(title=title, interval=interval, axis=axis, show_axis=show_axis,
                         show_displacement_vectors=False)

        time_steps = self.time_steps

        class PauseAnimation:
            def __init__(self):
                self.ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=100)
                self.paused = False

                fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

            def toggle_pause(self, *args, **kwargs):
                if self.paused:
                    self.ani.resume()
                else:
                    self.ani.pause()
                self.paused = not self.paused

            def save(self, filename='animation.gif', writer='imagemagick', fps=10):
                self.ani.save(filename, writer=writer, fps=fps)

        ani = PauseAnimation()
        return ani

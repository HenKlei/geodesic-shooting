import numpy as np
from numbers import Number
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skimage

from geodesic_shooting.core.base import BaseFunction, BaseTimeDependentFunction
from geodesic_shooting.utils import grid


class ScalarFunction(BaseFunction):
    """Class that represents a scalar-valued function (i.e. an image)."""
    def __init__(self, spatial_shape=(), data=None):
        """Constructor.

        Parameters
        ----------
        spatial_shape
            Spatial shape of the underlying domain, i.e. number of components
            in each coordinate direction.
        data
            numpy-array containing the values of the `ScalarFunction`. If `None`,
            one has to provide `spatial_shape` and a numpy-array containing zeros
            with shape `spatial_shape` is created as data. If not `None`, either
            `spatial_shape` is the empty tuple or the shape of `data` fits to
            the provided `spatial_shape`.
        """
        super().__init__(spatial_shape, data)

    def _compute_spatial_shape(self, spatial_shape, data):
        if data is None:
            return spatial_shape
        return data.shape

    def _compute_full_shape(self):
        return self.spatial_shape

    def plot(self, title="", colorbar=True, axis=None, show_axis=True, figsize=(10, 10), extent=(0., 1., 0., 1.),
             vmin=None, vmax=None, show_restriction_boundary=True, restriction=np.s_[...]):
        """Plots the `ScalarFunction` using `matplotlib`.

        Parameters
        ----------
        title
            The title of the plot.
        colorbar
            Determines whether or not to show a colorbar (only used if `axis=None`).
        axis
            If not `None`, the function is plotted on the provided axis.
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.
        extent
            Determines the left, right, bottom, and top coordinates of the plot.
            Only used in the 2-dimensional case.
        vmin
            Minimum value defining the data range of the colorbar.
        vmax
            Maximum value defining the data range of the colorbar.

        Returns
        -------
        If `axis` is `None`, the created figure, the axis and the values returned by the
        plot function (can be used to create colorbars) are returned, otherwise the axis
        and the values returned by the plot function (can be used to create colorbars) are
        returned.
        """
        assert self.dim in {1, 2, 3}

        created_figure = False
        if not axis:
            created_figure = True
            if self.dim == 3:
                fig, axis = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': '3d'})
            else:
                fig, axis = plt.subplots(1, 1, figsize=figsize)

        if self.dim == 1:
            vals = axis.plot(self.to_numpy())
        elif self.dim == 2:
            image = np.copy(self.to_numpy().transpose())
            if show_restriction_boundary and restriction != np.s_[...]:
                label_img = np.zeros_like(image, dtype=bool)
                label_img[restriction] = True
                mask = skimage.segmentation.find_boundaries(label_img, mode='outer')
                image[mask] = np.nan
            vals = axis.imshow(image, origin='lower', extent=extent, vmin=vmin, vmax=vmax, aspect=None)
        elif self.dim == 3:
            identity_grid = grid.coordinate_grid(self.spatial_shape).to_numpy()
            vals = axis.scatter(identity_grid[..., 0].flatten(),
                                identity_grid[..., 1].flatten(),
                                identity_grid[..., 2].flatten(), c=self.to_numpy().flatten(), vmin=vmin, vmax=vmax)

        axis.set_title(title)
        if not show_axis:
            axis.set_axis_off()

        if created_figure:
            if colorbar:
                if not isinstance(vals, list):
                    fig.colorbar(vals, ax=axis)
            return fig, axis, vals
        return axis, vals

    def save(self, filepath, title="", colorbar=True, figsize=(10, 10), extent=(0., 1., 0., 1.), dpi=100,
             show_axis=True, show_restriction_boundary=True, restriction=np.s_[...]):
        """Saves the plot of the `ScalarFunction` produced by the `plot`-function.

        Parameters
        ----------
        filepath
            Path to save the plot to.
        title
            Title of the plot.
        colorbar
            Determines whether or not to show a colorbar.
        extent
            Determines the left, right, bottom, and top coordinates of the plot.
            Only used in the 2-dimensional case.
        dpi
            The resolution in dots per inch.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
            for more details.
        """
        try:
            fig, _, _ = self.plot(title=title, colorbar=colorbar, axis=None, show_axis=show_axis, figsize=figsize,
                                  extent=extent, show_restriction_boundary=show_restriction_boundary,
                                  restriction=restriction)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

    def abs(self):
        """Returns new `ScalarFunction` containing absolute values of the elements.

        Returns
        -------
        `ScalarFunction` similar to the original one, but with absolute value in each element.
        """
        c = self.copy()
        c._data = np.abs(c._data)
        return c

    def __add__(self, other):
        assert ((isinstance(other, ScalarFunction) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and other.shape == self.full_shape))
        result = self.copy()
        if isinstance(other, ScalarFunction):
            result._data += other._data
        else:
            result._data += other
        return result

    __radd__ = __add__

    def __iadd__(self, other):
        assert ((isinstance(other, ScalarFunction) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and other.shape == self.full_shape))
        if isinstance(other, ScalarFunction):
            self._data = self._data + other._data
        else:
            self._data = self._data + other
        return self

    def __sub__(self, other):
        assert ((isinstance(other, ScalarFunction) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and other.shape == self.full_shape))
        result = self.copy()
        if isinstance(other, ScalarFunction):
            result._data -= other._data
        else:
            result._data -= other
        return result

    def __isub__(self, other):
        assert ((isinstance(other, ScalarFunction) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and other.shape == self.full_shape))
        if isinstance(other, ScalarFunction):
            self._data = self._data - other._data
        else:
            self._data = self._data - other
        return self

    def __neg__(self):
        return -1. * self

    def __mul__(self, other):
        assert isinstance(other, Number)
        result = self.copy()
        result._data *= other
        return result

    __rmul__ = __mul__

    def __imul__(self, other):
        assert isinstance(other, Number)
        self._data = other * self._data
        return self

    def __truediv__(self, other):
        assert isinstance(other, Number)
        result = self.copy()
        result._data /= other
        return result

    def __itruediv__(self, other):
        assert isinstance(other, Number)
        self._data = self._data / other
        return self

    def __pow__(self, exp):
        result = self.copy()
        result._data = result._data**exp
        return result


class TimeDependentScalarFunction(BaseTimeDependentFunction):
    def __init__(self, spatial_shape=(), time_steps=1, data=None):
        super().__init__(spatial_shape, time_steps, data, time_independent_type=ScalarFunction)

    def _compute_full_shape(self):
        return (self.time_steps, *self.spatial_shape)

    def animate(self, title="", colorbar=True, show_axis=True, figsize=(10, 10), extent=(0., 1., 0., 1.)):
        """Animates the `TimeDependentScalarFunction` using the `plot`-function of `ScalarFunction`.

        Parameters
        ----------
        title
            The title of the plot.
        colorbar
            Determines whether or not to show a colorbar.
        show_axis
            Determines whether or not to show the axes.
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.
        extent
            Determines the left, right, bottom, and top coordinates of the plot.
            Only used in the 2-dimensional case.

        Returns
        -------
        The animation object.
        """
        if self.dim == 3:
            fig, axis = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': '3d'})
        else:
            fig, axis = plt.subplots(1, 1, figsize=figsize)

        if not show_axis:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        zmin = np.min(self.to_numpy())
        zmax = np.max(self.to_numpy())
        _, vals = self[0].plot(title=title, axis=axis, extent=extent, show_axis=show_axis, vmin=zmin, vmax=zmax)
        if colorbar:
            fig.colorbar(vals, ax=axis)

        def update(i):
            axis.clear()
            _, vals = self[i].plot(title=title, axis=axis, extent=extent, show_axis=show_axis, vmin=zmin, vmax=zmax)

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

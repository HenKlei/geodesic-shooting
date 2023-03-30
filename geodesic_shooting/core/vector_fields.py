import numpy as np
from numbers import Number
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from pyevtk.vtk import VtkFile, VtkRectilinearGrid

from geodesic_shooting.core.base import BaseFunction, BaseTimeDependentFunction
from geodesic_shooting.core import (Diffeomorphism, TimeDependentDiffeomorphism,
                                    ScalarFunction, TimeDependentScalarFunction)
from geodesic_shooting.utils import sampler, grid
from geodesic_shooting.utils.helper_functions import lincomb


class VectorField(BaseFunction):
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

    def get_magnitude(self):
        """Computes the magnitude of the `VectorField` as a `ScalarFunction`.

        Returns
        -------
        The magnitude as a `ScalarFunction`.
        """
        return ScalarFunction(data=np.linalg.norm(self.to_numpy(), axis=-1))

    def get_component_as_function(self, component=0):
        """Provides the specified component of the `VectorField` as a `ScalarFunction`.

        Parameters
        ----------
        component
            Index of the component to return.

        Returns
        -------
        The component as a `ScalarFunction`.
        """
        assert isinstance(component, int) and 0 <= component < self.dim
        return ScalarFunction(data=self.to_numpy()[..., component])

    def get_divergence(self, return_gradient=False):
        """Computes the divergence of the `VectorField`.

        Parameters
        ----------
        return_gradient
            Determines whether or not to also return the gradient (Jacobian) of the `VectorField`.

        Returns
        -------
        The divergence as a `ScalarFunction`. If `return_gradient` is `True`, also the gradient
        is returned.
        """
        grad = self.grad
        div = ScalarFunction(data=np.sum(np.array([grad[..., d, d] for d in range(self.dim)]), axis=0))
        if return_gradient:
            return div, grad
        return div

    div = property(get_divergence)

    def plot(self, title="", interval=1, color_length=False, colorbar=True, vmin=None, vmax=None, show_axis=True,
             scale=None, axis=None, figsize=(10, 10), zorder=1):
        """Plots the `VectorField` using `matplotlib`'s `quiver` function.

        Parameters
        ----------
        title
            The title of the plot.
        interval
            Interval in which to sample.
        color_length
            Determines whether or not to show the lengths of the vectors using
            different colors.
        colorbar
            Determines whether or not to show a colorbar.
            Only used if `color_length` is `True`.
        vmin
            Minimum value defining the data range of the colorbar.
        vmax
            Maximum value defining the data range of the colorbar.
        show_axis
            Determines whether or not to show the axes.
        scale
            Factor used for scaling the arrows in the `quiver`-plot.
            If `None`, a default auto-scaling from `matplotlib` is applied.
            For realistic arrow lengths without scaling, a value of `scale=1.`
            has to be used.
        axis
            If not `None`, the function is plotted on the provided axis.
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.
        zorder
            Determines the ordering of the plots on the axis.

        Returns
        -------
        If `axis` is None, the created figure and the axis are returned,
        otherwise only the altered axis is returned.
        """
        assert self.dim in {1, 2, 3}

        created_figure = False
        if not axis:
            created_figure = True
            if self.dim == 3:
                fig, axis = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': '3d'})
            else:
                fig, axis = plt.subplots(1, 1, figsize=figsize)

        if not show_axis:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if np.max(np.linalg.norm(self.to_numpy(), axis=-1)) < 1e-10:
            scale = 1

        identity_grid = grid.coordinate_grid(self.spatial_shape)

        if color_length:
            colors = np.linalg.norm(self.to_numpy(), axis=-1)
            clim = None
            if vmin and vmax:
                clim = (vmin, vmax)
            if self.dim == 1:
                vals = axis.quiver(identity_grid[::interval, 0] / self.spatial_shape[0], np.zeros(self.spatial_shape),
                                   self[::interval, 0], np.zeros(self.spatial_shape), colors,
                                   scale_units='xy', units='xy', angles='xy', scale=scale, zorder=zorder, clim=clim)
            elif self.dim == 2:
                vals = axis.quiver(identity_grid[::interval, ::interval, 0] / self.spatial_shape[0],
                                   identity_grid[::interval, ::interval, 1] / self.spatial_shape[1],
                                   self[::interval, ::interval, 0], self[::interval, ::interval, 1], colors,
                                   scale_units='xy', units='xy', angles='xy', scale=scale, zorder=zorder, clim=clim)
            elif self.dim == 3:
                vals = axis.quiver(identity_grid[::interval, ::interval, ::interval, 0] / self.spatial_shape[0],
                                   identity_grid[::interval, ::interval, ::interval, 1] / self.spatial_shape[1],
                                   identity_grid[::interval, ::interval, ::interval, 2] / self.spatial_shape[2],
                                   self[::interval, ::interval, ::interval, 0],
                                   self[::interval, ::interval, ::interval, 1],
                                   self[::interval, ::interval, ::interval, 2],
                                   colors=colors, zorder=zorder, clim=clim)
            if colorbar and created_figure:
                fig.colorbar(vals, ax=axis)
        else:
            if self.dim == 1:
                axis.quiver(identity_grid[::interval, 0] / self.spatial_shape[0], np.zeros(self.spatial_shape),
                            self[::interval, 0], np.zeros(self.spatial_shape),
                            scale_units='xy', units='xy', angles='xy', scale=scale, zorder=zorder)
            elif self.dim == 2:
                axis.quiver(identity_grid[::interval, ::interval, 0] / self.spatial_shape[0],
                            identity_grid[::interval, ::interval, 1] / self.spatial_shape[1],
                            self[::interval, ::interval, 0], self[::interval, ::interval, 1],
                            scale_units='xy', units='xy', angles='xy', scale=scale, zorder=zorder)
            elif self.dim == 3:
                axis.quiver(identity_grid[::interval, ::interval, ::interval, 0] / self.spatial_shape[0],
                            identity_grid[::interval, ::interval, ::interval, 1] / self.spatial_shape[1],
                            identity_grid[::interval, ::interval, ::interval, 2] / self.spatial_shape[2],
                            self[::interval, ::interval, ::interval, 0],
                            self[::interval, ::interval, ::interval, 1],
                            self[::interval, ::interval, ::interval, 2],
                            zorder=zorder)

        if created_figure:
            return fig, axis
        if color_length and colorbar and not created_figure:
            return axis, vals
        return axis

    def plot_streamlines(self, title="", density=1, color_length=False, show_axis=True, axis=None, figsize=(10, 10),
                         zorder=1, integration_direction='forward'):
        """Plots the `VectorField` using `matplotlib`'s `quiver` function.

        Parameters
        ----------
        title
            The title of the plot.
        density
            Determines the density of the streamlines.
        color_length
            Determines whether or not to show the lengths of the vectors using
            different colors.
        show_axis
            Determines whether or not to show the axes.
        axis
            If not `None`, the function is plotted on the provided axis.
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.
        zorder
            Determines the ordering of the plots on the axis.
        integration_direction
            The direction in which to integrate the streamlines.

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

        if not show_axis:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        xs = np.arange(self.spatial_shape[0]) / self.spatial_shape[0]
        ys = np.arange(self.spatial_shape[1]) / self.spatial_shape[1]
        if color_length:
            colors = np.linalg.norm(self.to_numpy(), axis=-1).T
            axis.streamplot(xs, ys, self[..., 0].T, self[..., 1].T,
                            density=density, color=colors, zorder=zorder, integration_direction=integration_direction)
        else:
            axis.streamplot(xs, ys, self[..., 0].T, self[..., 1].T,
                            density=density, zorder=zorder, integration_direction=integration_direction)

        if created_figure:
            return fig, axis
        return axis

    def plot_as_warpgrid(self, title="", interval=1, show_axis=True, show_identity_grid=True, axis=None,
                         figsize=(10, 10), show_displacement_vectors=False, color_length=False):
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
        dist_x, dist_y = grid_x + dist_x * self.spatial_shape[0], grid_y + dist_y * self.spatial_shape[1]
        plot_grid(dist_x, dist_y, color="C0")

        if show_displacement_vectors:
            if color_length:
                colors = np.linalg.norm(self.to_numpy(), axis=-1)
                axis.quiver(identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1],
                            self[::interval, ::interval, 0] * self.spatial_shape[0],
                            self[::interval, ::interval, 1] * self.spatial_shape[1],
                            colors, scale_units='xy', units='xy', angles='xy', scale=1, zorder=2)
            else:
                axis.quiver(identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1],
                            self[::interval, ::interval, 0] * self.spatial_shape[0],
                            self[::interval, ::interval, 1] * self.spatial_shape[1],
                            scale_units='xy', units='xy', angles='xy', scale=1, zorder=2)

        if not show_axis:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if created_figure:
            return fig, axis
        return axis

    def save(self, filepath, dpi=100, plot_type='default',
             plot_args={'title': '', 'interval': 1, 'color_length': False, 'show_axis': True, 'scale': None,
                        'axis': None, 'figsize': (20, 20)}):
        """Saves the plot of the `VectorField` produced by the `plot`-function.

        Parameters
        ----------
        filepath
            Path to save the plot to.
        dpi
            The resolution in dots per inch.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
            for more details.
        plot_type
            String determining the type of the plot, possible options are `default`
            (corresponds to a quiver-plot), `streamlines` and `warpgrid`.
        plot_args
            Dictionary of optional arguments that can be passed to the plot function chosen
            as `plot_type`.
        """
        assert plot_type in {'default', 'streamlines', 'warpgrid'}
        if 'axis' in plot_args:
            assert plot_args['axis'] is None

        try:
            if plot_type == 'default':
                fig, _ = self.plot(**plot_args)
            elif plot_type == 'streamlines':
                fig, _ = self.plot_streamlines(**plot_args)
            elif plot_type == 'warpgrid':
                fig, _ = self.plot_as_warpgrid(**plot_args)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

    def save_tikz(self, filepath, title="", interval=1, scale=1.):
        """Saves the plot of the `VectorField` as a tikz file.

        Parameters
        ----------
        filepath
            Path to save the plot to.
        title
            Title of the plot.
        interval
            Interval in which to sample.
        scale
            Factor used for scaling the arrows in the plot.
        """
        assert isinstance(interval, int)
        assert isinstance(scale, float) or isinstance(scale, int)

        try:
            with open(filepath, "w") as tikz_file:
                tikz_file.write("\\documentclass{standalone}\n\n"
                                "\\usepackage{tikz}\n"
                                "\\usepackage{pgfplots}\n"
                                "\\begin{document}\n\n")
                tikz_file.write("\t\\begin{tikzpicture}\n"
                                "\t\t\\begin{axis}[tick align=outside, tick pos=left, "
                                "title={"
                                f"{title}"
                                "}, xmin=0, xmax=1, "
                                "xtick style={color=black}, ymin=0, ymax=1, ytick style={color=black}, -latex]\n")
                x = grid.coordinate_grid(self.spatial_shape).to_numpy()
                for pos_x, disp_x in zip(x[::interval], self[::interval]):
                    for pos, disp in zip(pos_x[::interval], disp_x[::interval]):
                        tikz_file.write(f"\t\t\t\\draw (axis cs:{pos[0] / self.spatial_shape[0]}, "
                                        f"{pos[1] / self.spatial_shape[1]}) "
                                        f"-- (axis cs:{(pos[0] + disp[0] * scale) / self.spatial_shape[0]}, "
                                        f"{(pos[1] + disp[1] * scale) / self.spatial_shape[1]});\n")
                tikz_file.write("\t\t\\end{axis}\n"
                                "\t\\end{tikzpicture}\n"
                                "\\end{document}\n")
        except Exception:
            pass

    def save_vtk(self, filepath, data_label="Vector field"):
        """Saves the `VectorField` as a vtk file.

        Parameters
        ----------
        filepath
            Path to save the plot to (without file extension).
        data_label
            Label for the vector field data.
        """
        assert self.dim in [2, 3]

        if self.dim == 2:
            nx, ny = self.spatial_shape
            nz = 1
        elif self.dim == 3:
            nx, ny, nz = self.spatial_shape
        lx, ly, lz = 1.0, 1.0, 1.0
        x = np.linspace(0, lx, nx, dtype="float64")
        y = np.linspace(0, ly, ny, dtype="float64")
        if self.dim == 2:
            z = np.linspace(0, lz, nz-1, dtype="float64")
        elif self.dim == 3:
            z = np.linspace(0, lz, nz, dtype="float64")
        start, end = (0, 0, 0), (nx-1, ny-1, nz-1)

        w = VtkFile(filepath, VtkRectilinearGrid)
        w.openGrid(start=start, end=end)
        w.openPiece(start=start, end=end)

        vx = np.array(self.to_numpy()[..., 0], order='F')
        vy = np.array(self.to_numpy()[..., 1], order='F')
        if self.dim == 2:
            vx = vx[..., np.newaxis]
            vy = vy[..., np.newaxis]
            vz = np.zeros_like(vx, order='F')
        elif self.dim == 3:
            vz = np.array(self.to_numpy()[..., 2], order='F')
        w.openData("Point", vectors=data_label)
        w.addData(data_label, (vx, vy, vz))
        w.closeData("Point")

        w.openElement("Coordinates")
        w.addData("x_coordinates", x)
        w.addData("y_coordinates", y)
        w.addData("z_coordinates", z)
        w.closeElement("Coordinates")

        w.closePiece()
        w.closeGrid()

        w.appendData(data=(vx, vy, vz))
        w.appendData(x).appendData(y).appendData(z)
        w.save()

    def __add__(self, other):
        assert ((isinstance(other, BaseFunction) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and (other.shape == self.full_shape
                                                       or other.shape == (self.dim, ))))
        result = self.copy()
        if isinstance(other, BaseFunction):
            result._data += other._data
        else:
            result._data += other
        return result

    __radd__ = __add__

    def __iadd__(self, other):
        assert ((isinstance(other, VectorField) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and (other.shape == self.full_shape
                                                       or other.shape == (self.dim, ))))
        if isinstance(other, VectorField):
            self._data = self._data + other._data
        else:
            self._data = self._data + other
        return self

    def __sub__(self, other):
        assert ((isinstance(other, VectorField) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and (other.shape == self.full_shape
                                                       or other.shape == (self.dim, ))))
        result = self.copy()
        if isinstance(other, VectorField):
            result._data -= other._data
        else:
            result._data -= other
        return result

    def __isub__(self, other):
        assert ((isinstance(other, VectorField) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and (other.shape == self.full_shape
                                                       or other.shape == (self.dim, ))))
        if isinstance(other, VectorField):
            self._data = self._data - other._data
        else:
            self._data = self._data - other
        return self

    def __neg__(self):
        return -1. * self

    def __mul__(self, other):
        result = self.copy()
        result._data *= other
        return result

    __rmul__ = __mul__

    def __imul__(self, other):
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


class TimeDependentVectorField(BaseTimeDependentFunction):
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
        super().__init__(spatial_shape, time_steps, data, time_independent_type=VectorField)

    def get_magnitude_series(self):
        """Computes the evolution of the magnitude as a `TimeDependentScalarFunction`.

        Returns
        -------
        The magnitude as a `TimeDependentScalarFunction`.
        """
        return TimeDependentScalarFunction(data=[v.get_magnitude() for v in self])

    def get_component_as_function_series(self, component=0):
        """Provides the evolution of the specified component as a `TimeDependentScalarFunction`.

        Parameters
        ----------
        component
            Index of the component to return.

        Returns
        -------
        The component as a `TimeDependentScalarFunction`.
        """
        return TimeDependentScalarFunction(data=[v.get_component_as_function(component) for v in self])

    @property
    def average(self):
        """Computes the average `VectorField` over time.

        Returns
        -------
        Average of the `VectorField`s.
        """
        return lincomb(self, np.ones(len(self)) / len(self))

    def integrate(self, sampler_options={'order': 1, 'mode': 'edge'}, get_time_dependent_diffeomorphism=False):
        """Integrate vector field with respect to time.

        Parameters
        ----------
        sampler_options
            Additional options to pass to the sampler.

        Returns
        -------
        Vector field containing the diffeomorphism originating from integrating the
        time-dependent vector field with respect to time.
        """
        # initial transformation is the identity mapping
        diffeomorphisms = [grid.identity_diffeomorphism(self.spatial_shape)]

        # perform integration with respect to time
        for t in range(self.time_steps):
            d = diffeomorphisms[-1] + sampler.sample(self[t], diffeomorphisms[-1],
                                                     sampler_options=sampler_options) / self.time_steps
            diffeomorphisms.append(Diffeomorphism(data=d))

        if get_time_dependent_diffeomorphism:
            return TimeDependentDiffeomorphism(data=diffeomorphisms)
        return diffeomorphisms[-1]

    def integrate_backward(self, sampler_options={'order': 1, 'mode': 'edge'}, get_time_dependent_diffeomorphism=False):
        """Integrate vector field backward with respect to time.

        Parameters
        ----------
        sampler_options
            Additional options to pass to the sampler.
        get_time_dependent_diffeomorphism
            Determines whether or not to return the `TimeDependentDiffeomorphism` or only
            the final `Diffeomorphism`.

        Returns
        -------
        Vector field containing the inverse diffeomorphism originating from integrating
        the time-dependent vector field backward with respect to time.
        """
        # initial transformation is the identity mapping
        diffeomorphisms = [grid.identity_diffeomorphism(self.spatial_shape)]

        # perform integration backwards with respect to time
        for t in range(self.time_steps-1, -1, -1):
            d = VectorField(data=diffeomorphisms[-1]) - (sampler.sample(self[t], diffeomorphisms[-1],
                                                                        sampler_options=sampler_options)
                                                         / self.time_steps)
            diffeomorphisms.append(Diffeomorphism(data=d))

        if get_time_dependent_diffeomorphism:
            return TimeDependentDiffeomorphism(data=diffeomorphisms)
        return diffeomorphisms[-1]

    def animate(self, title="", interval=1, color_length=False, colorbar=True, scale=None, show_axis=True,
                figsize=(10, 10)):
        """Animates the `TimeDependentVectorField` using the `plot`-function of `VectorField`.

        Parameters
        ----------
        title
            The title of the plot.
        interval
            Interval in which to sample.
        color_length
            Determines whether or not to show the lengths of the vectors using
            different colors.
        colorbar
            Determines whether or not to show a colorbar.
            Only used if `color_length` is `True`.
        scale
            Factor used for scaling the arrows in the `quiver`-plot.
            If `None`, a default auto-scaling from `matplotlib` is applied.
            For realistic arrow lengths without scaling, a value of `scale=1.`
            has to be used.
        show_axis
            Determines whether or not to show the axes.
        figsize
            Width and height of the figure in inches.
            Only used if `axis` is `None` and a new figure is created.

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

        if color_length and colorbar:
            vmin = np.min(self.get_magnitude_series().to_numpy())
            vmax = np.max(self.get_magnitude_series().to_numpy())
            _, vals = self[0].plot(title=title, interval=interval, color_length=color_length, colorbar=True,
                                   show_axis=show_axis, vmin=vmin, vmax=vmax, scale=scale, axis=axis)
            fig.colorbar(vals)

        def update(i):
            axis.clear()
            self[i].plot(title=title, interval=interval, color_length=color_length, colorbar=False,
                         show_axis=show_axis, scale=scale, axis=axis)

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

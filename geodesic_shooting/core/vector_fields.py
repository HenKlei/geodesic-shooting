import numpy as np
from numbers import Number
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils import sampler, grid
import geodesic_shooting.utils.grad as grad
from geodesic_shooting.utils.helper_functions import lincomb


class VectorField:
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
        if isinstance(data, VectorField):
            data = data._data

        if data is None:
            assert spatial_shape != ()
        else:
            if spatial_shape != ():
                assert spatial_shape == data.shape[0:-1]
            else:
                spatial_shape = data.shape[0:-1]

        self.spatial_shape = spatial_shape
        self.dim = len(self.spatial_shape)
        self.full_shape = (*self.spatial_shape, self.dim)

        if data is None:
            data = np.zeros(self.full_shape)
        assert len(spatial_shape) == data.shape[-1]
        self._data = data
        assert self._data.shape == self.full_shape

    @property
    def grad(self):
        """Computes the (discrete, approximate) gradient/Jacobian using finite differences.

        Returns
        -------
        Finite difference approximation of the gradient/Jacobian.
        """
        return grad.finite_difference(self)

    def to_numpy(self, shape=None):
        """Returns the `VectorField` represented as a numpy-array.

        Parameters
        ----------
        shape
            If not `None`, the numpy-array is reshaped according to `shape`.

        Returns
        -------
        Numpy-array containing the entries of the `VectorField`.
        """
        if shape:
            return self._data.reshape(shape)
        return self._data

    def flatten(self):
        """Returns the `VectorField` represented as a flattened numpy-array.

        Returns
        -------
        Flattened numpy-array containing the entries of the `VectorField`.
        """
        return self.to_numpy().flatten()

    def push_forward(self, flow, sampler_options={'order': 1, 'mode': 'edge'}):
        """Pushes forward the `VectorField` along a flow.

        Parameters
        ----------
        flow
            `VectorField` containing the flow according to which to push the input forward.
        sampler_options
            Additional options passed to the `warp`-function, see
            https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp.

        Returns
        -------
        `VectorField` of the forward-pushed function.
        """
        return sampler.sample(self, flow, sampler_options=sampler_options)

    def push_backward(self, flow, sampler_options={'order': 1, 'mode': 'edge'}):
        """Pushes backward the `VectorField` along a flow.

        Parameters
        ----------
        flow
            `VectorField` containing the flow according to which to push the input backward.
        sampler_options
            Additional options passed to the `warp`-function, see
            https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp.

        Returns
        -------
        `VectorField` of the backward-pushed function.
        """
        return sampler.sample_inverse(self, flow, sampler_options=sampler_options)

    def plot(self, title="", interval=1, color_length=False, show_axis=False, scale=None, axis=None, zorder=1):
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
        show_axis
            Determines whether or not to show the axes.
        scale
            Factor used for scaling the arrows in the `quiver`-plot.
            If `None`, a default auto-scaling from `matplotlib` is applied.
            For realistic arrow lengths without scaling, a value of `scale=1.`
            has to be used.
        axis
            If not `None`, the function is plotted on the provided axis.
        zorder
            Determines the ordering of the plots on the axis.

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

        if show_axis is False:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if np.max(np.linalg.norm(self.to_numpy(), axis=-1)) < 1e-10:
            scale = 1

        identity_grid = grid.coordinate_grid(self.spatial_shape)

        if color_length:
            colors = np.linalg.norm(self.to_numpy(), axis=-1)
            axis.quiver(identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1],
                        self[::interval, ::interval, 0], self[::interval, ::interval, 1], colors,
                        scale_units='xy', units='xy', angles='xy', scale=scale, zorder=zorder)
        else:
            axis.quiver(identity_grid[::interval, ::interval, 0], identity_grid[::interval, ::interval, 1],
                        self[::interval, ::interval, 0], self[::interval, ::interval, 1],
                        scale_units='xy', units='xy', angles='xy', scale=scale, zorder=zorder)

        if created_figure:
            return fig, axis
        return axis

    def plot_streamlines(self, title="", density=1, color_length=False, show_axis=False, axis=None,
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
        zorder
            Determines the ordering of the plots on the axis.
        integration_direction
            The direction in which to integrate the streamlines.

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

        if show_axis is False:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if color_length:
            colors = np.linalg.norm(self.to_numpy(), axis=-1).T
            xs = np.arange(self.spatial_shape[0])
            ys = np.arange(self.spatial_shape[1])
            axis.streamplot(xs, ys, self[..., 0].T, self[..., 1].T,
                            density=density, color=colors, zorder=zorder, integration_direction=integration_direction)
        else:
            xs = np.arange(self.spatial_shape[0])
            ys = np.arange(self.spatial_shape[1])
            axis.streamplot(xs, ys, self[..., 0].T, self[..., 1].T,
                            density=density, zorder=zorder, integration_direction=integration_direction)

        if created_figure:
            return fig, axis
        return axis

    def plot_as_warpgrid(self, title="", interval=1, show_axis=False, show_identity_grid=True, axis=None,
                         show_displacement_vectors=False):
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
        grid_x, grid_y = identity_grid[..., 0], identity_grid[..., 1]
        if show_identity_grid:
            plot_grid(grid_x, grid_y, color="lightgrey")

        distx, disty = grid_x + self[..., 0], grid_y + self[..., 1]
        plot_grid(distx, disty, color="C0")

        if show_displacement_vectors:
            self.plot(scale=1., axis=axis, zorder=2)

        if show_axis is False:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        if created_figure:
            return fig, axis
        return axis

    def save(self, filepath, title="", interval=1, color_length=False, show_axis=False, scale=None):
        """Saves the plot of the `VectorField` produced by the `plot`-function.

        Parameters
        ----------
        filepath
            Path to save the plot to.
        title
            Title of the plot.
        interval
            Interval in which to sample.
        color_length
            Determines whether or not to show the lengths of the vectors using
            different colors.
        show_axis
            Determines whether or not to show the axes.
        scale
            Factor used for scaling the arrows in the `quiver`-plot.
            If `None`, a default auto-scaling from `matplotlib` is applied.
            For realistic arrow lengths without scaling, a value of `scale=1.`
            has to be used.
        """
        try:
            fig, _ = self.plot(title=title, interval=interval, color_length=color_length,
                               show_axis=show_axis, scale=scale, axis=None)
            fig.savefig(filepath)
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
                        tikz_file.write(f"\t\t\t\\draw (axis cs:{pos[0]/self.spatial_shape[0]}, "
                                        f"{pos[1]/self.spatial_shape[1]}) "
                                        f"-- (axis cs:{(pos[0]+disp[0]*scale)/self.spatial_shape[0]}, "
                                        f"{(pos[1]+disp[1]*scale)/self.spatial_shape[1]});\n")
                tikz_file.write("\t\t\\end{axis}\n"
                                "\t\\end{tikzpicture}\n"
                                "\\end{document}\n")
        except Exception:
            pass

    def get_norm(self, product_operator=None, order=None, restriction=np.s_[...]):
        """Computes the norm of the `VectorField`.

        Remark: If `order=None` and `self.dim >= 2`, the 2-norm of `self.to_numpy().ravel()`
        is returned, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.

        Parameters
        ----------
        product_operator
            Operator with respect to which to compute the norm. If `None`, the standard l2-inner
            product is used.
        order
            Order of the norm,
            see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.
        restriction
            Slice that can be used to restrict the domain on which to compute the norm.

        Returns
        -------
        The norm of the `VectorField`.
        """
        if product_operator:
            apply_product_operator = product_operator(self).to_numpy()[restriction]
            return np.sqrt(apply_product_operator.flatten().dot(self.to_numpy()[restriction].flatten()))
        else:
            return np.linalg.norm(self.to_numpy()[restriction].flatten(), ord=order)

    norm = property(get_norm)

    def copy(self):
        """Returns a deepcopy of the `VectorField`.

        Returns
        -------
        Deepcopy of the whole `VectorField`.
        """
        return deepcopy(self)

    def __add__(self, other):
        assert ((isinstance(other, VectorField) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and (other.shape == self.full_shape
                                                       or other.shape == (self.dim, ))))
        result = self.copy()
        if isinstance(other, VectorField):
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

    def __eq__(self, other):
        if isinstance(other, VectorField) and (other.to_numpy() == self.to_numpy()).all():
            return True
        return False

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, val):
        if isinstance(val, VectorField) or isinstance(val, ScalarFunction):
            self._data[index] = val._data
        else:
            self._data[index] = val
        assert self._data.shape == self.full_shape

    def __str__(self):
        return str(self.to_numpy())


class TimeDependentVectorField:
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
        assert isinstance(time_steps, int) and time_steps > 0
        self.spatial_shape = spatial_shape
        self.dim = len(self.spatial_shape)
        self.time_steps = time_steps
        self.full_shape = (self.time_steps, *self.spatial_shape, self.dim)

        self._data = []
        if data is None:
            for _ in range(self.time_steps):
                self._data.append(VectorField(self.spatial_shape))
        else:
            assert ((isinstance(data, np.ndarray) and data.shape == self.full_shape)
                    or (isinstance(data, TimeDependentVectorField) and data.full_shape == self.full_shape)
                    or (isinstance(data, list) and all(isinstance(d, VectorField) for d in data)))
            for elem in data:
                if isinstance(elem, VectorField):
                    self._data.append(elem.copy())
                else:
                    self._data.append(VectorField(self.spatial_shape, data=elem))

    @property
    def average(self):
        """Computes the average `VectorField` over time.

        Returns
        -------
        Average of the `VectorField`s.
        """
        return lincomb(self, np.ones(len(self)) / len(self))

    def integrate(self, time_steps=30, sampler_options={'order': 1, 'mode': 'edge'}):
        """Integrate vector field with respect to time.

        Parameters
        ----------
        time_steps
            Number of time steps performed during integration.
        sampler_options
            Additional options to pass to the sampler.

        Returns
        -------
        Vector field containing the diffeomorphism originating from integrating the
        time-dependent vector field with respect to time.
        """
        # create identity grid
        identity_grid = grid.coordinate_grid(self.spatial_shape)

        # initial transformation is the identity mapping
        diffeomorphism = identity_grid.copy()

        # perform integration with respect to time
        for t in range(self.time_steps):
            diffeomorphism += sampler.sample(self[t], diffeomorphism,
                                             sampler_options=sampler_options) / self.time_steps

        return diffeomorphism

    def to_numpy(self, shape=None):
        """Returns the `TimeDependentVectorField` represented as a numpy-array.

        Parameters
        ----------
        shape
            If not `None`, the numpy-array is reshaped according to `shape`.

        Returns
        -------
        Numpy-array containing the entries of the `TimeDependentVectorField`.
        """
        result = np.array([a.to_numpy() for a in self._data])
        assert result.shape == self.full_shape
        if shape:
            return result.reshape(shape)
        return result

    def animate(self, title="", interval=1, scale=None, show_axis=False):
        """Animates the time-dependent vector field.

        Parameters
        ----------
        title
            The title of the plot.
        interval
            Interval in which to sample.
        scale
            Factor used for scaling the arrows in the `quiver`-plot.
            If `None`, a default auto-scaling from `matplotlib` is applied.
            For realistic arrow lengths without scaling, a value of `scale=1.`
            has to be used.
        show_axis
            Determines whether or not to show the axes.

        Returns
        -------
        The animation object.
        """
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        if show_axis is False:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        def animate(i):
            axis.clear()
            self[i].plot(title=title, interval=interval, scale=scale, axis=axis)

        ani = animation.FuncAnimation(fig, animate, frames=self.time_steps, interval=100)
        return ani

    def get_norm(self, order=None, restriction=np.s_[...]):
        """Computes the norm of the `TimeDependentVectorField`.

        Remark: If `order=None` and `self.dim >= 2`, the 2-norm of `self.to_numpy().ravel()`
        is returned, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.

        Parameters
        ----------
        order
            Order of the norm,
            see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.
        restriction
            Slice that can be used to restrict the domain on which to compute the norm.

        Returns
        -------
        The norm of the `TimeDependentVectorField`.
        """
        return np.linalg.norm(self.to_numpy()[restriction].flatten(), ord=order)

    norm = property(get_norm)

    def __setitem__(self, index, val):
        assert isinstance(val, VectorField) and val.full_shape == (*self.spatial_shape, self.dim)
        assert ((isinstance(index, int) and 0 <= index < self.time_steps)
                or (isinstance(index, tuple) and len(index) == 1 and 0 <= index[0] < self.time_steps))
        if isinstance(index, tuple):
            self._data[index[0]] = val
        else:
            self._data[index] = val

    def __getitem__(self, index):
        assert isinstance(index, int) or isinstance(index, tuple)
        if isinstance(index, int):
            return self._data[index]
        return self._data[index[0]][index[1:]]

    def __str__(self):
        return str(self.to_numpy())

    def __len__(self):
        return self.time_steps

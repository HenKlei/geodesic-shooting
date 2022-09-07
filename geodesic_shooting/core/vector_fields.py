import numpy as np
from numbers import Number
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils import grid
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
        return self.to_numpy().flatten()

    def plot(self, title="", interval=1, show_axis=False, scale=None, axis=None):
        """Plots the `VectorField` using `matplotlib`'s `quiver` function.

        Parameters
        ----------
        title
            The title of the plot.
        interval
            Interval in which to sample.
        show_axis
            Determines whether or not to show the axes.
        scale
            Factor used for scaling the arrows in the `quiver`-plot.
            If `None`, a default auto-scaling from `matplotlib` is applied.
            For realistic arrow lengths without scaling, a value of `scale=1.`
            has to be used.
        axis
            If not `None`, the function is plotted on the provided axis.

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

        x = grid.coordinate_grid(self.spatial_shape).to_numpy()
        axis.quiver(x[::interval, ::interval, 0], x[::interval, ::interval, 1],
                    self[::interval, ::interval, 0], self[::interval, ::interval, 1],
                    units='xy', scale=scale)

        if created_figure:
            return fig, axis
        return axis

    def save(self, filepath, title=""):
        """Saves the plot of the `VectorField` produced by the `plot`-function.

        Parameters
        ----------
        filepath
            Path to save the plot to.
        title
            Title of the plot.
        """
        fig, _ = self.plot(title=title, axis=None)
        fig.savefig(filepath)
        plt.close(fig)

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


    def plot_as_warpgrid(self, title="", interval=1, show_axis=False, invert_yaxis=True, axis=None):
        """Plots the `VectorField` as a warpgrid using `matplotlib`.

        Parameters
        ----------
        title
            The title of the plot.
        interval
            Interval in which to sample.
        show_axis
            Determines whether or not to show the axes.
        invert_yaxis
            Determines whether or not to invert the vertical axis.
        axis
            If not `None`, the function is plotted on the provided axis.

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

        if invert_yaxis:
            axis.invert_yaxis()
        axis.set_aspect('equal')
        axis.set_title(title)

        for row in range(0, self.spatial_shape[0], interval):
            axis.plot(self[row, :, 0], self[row, :, 1], 'k')
        for col in range(0, self.spatial_shape[1], interval):
            axis.plot(self[:, col, 0], self[:, col, 1], 'k')

        if created_figure:
            return fig, axis
        return axis

    def get_norm(self, order=None, restriction=np.s_[...]):
        """Computes the norm of the `VectorField`.

        Remark: If `order=None` and `self.dim >= 2`, the 2-norm of `self.to_numpy().ravel()`
        is returned, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.

        Parameters
        ----------
        order
            Order of the norm,
            see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.

        Returns
        -------
        The norm of the `VectorField`.
        """
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
                or (isinstance(other, np.ndarray) and other.shape == self.full_shape))
        result = self.copy()
        if isinstance(other, VectorField):
            result._data += other._data
        else:
            result._data += other
        return result

    __radd__ = __add__

    def __iadd__(self, other):
        assert ((isinstance(other, VectorField) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and other.shape == self.full_shape))
        if isinstance(other, VectorField):
            self._data = self._data + other._data
        else:
            self._data = self._data + other
        return self

    def __sub__(self, other):
        assert ((isinstance(other, VectorField) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and other.shape == self.full_shape))
        result = self.copy()
        if isinstance(other, VectorField):
            result._data -= other._data
        else:
            result._data -= other
        return result

    def __isub__(self, other):
        assert ((isinstance(other, VectorField) and other.full_shape == self.full_shape)
                or (isinstance(other, np.ndarray) and other.shape == self.full_shape))
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
            If `None`, one has to provide `spatial_shape` and `time_steps` and a
            list containing for each time step a `VectorField` of zeros with shape
            `(*spatial_shape, dim)` is created as data, where `dim` is the dimension of
            the underlying domain (given as `len(spatial_shape)`).
            If not `None`, either a numpy-array or a `TimeDependentVectorField`
            has to be provided.
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
                    or (isinstance(data, TimeDependentVectorField) and data.full_shape == self.full_shape))
            for elem in data:
                if isinstance(elem, VectorField):
                    self._data.append(elem.copy())
                else:
                    self._data.append(VectorField(self.spatial_shape, data=elem))

    @property
    def average(self):
        return lincomb(self, np.ones(len(self)) / len(self))

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

    def get_norm(self, order=None):
        """Computes the norm of the `TimeDependentVectorField`.

        Remark: If `order=None` and `self.dim >= 2`, the 2-norm of `self.to_numpy().ravel()`
        is returned, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.

        Parameters
        ----------
        order
            Order of the norm,
            see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.

        Returns
        -------
        The norm of the `TimeDependentVectorField`.
        """
        return np.linalg.norm(self.to_numpy(), ord=order)

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

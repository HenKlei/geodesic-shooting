import numpy as np
from numbers import Number
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils import grid
import geodesic_shooting.utils.grad as grad


class VectorField:
    def __init__(self, spatial_shape=(), data=None):
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
        return grad.finite_difference(self)

    def plot(self, title="", interval=1, show_axis=False, scale=None, axis=None, show=True):
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
        assert self.dim == 2

        fig_created = False
        if not axis:
            fig_created = True
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

        if fig_created:
            if show:
                plt.show()
            return fig
        return axis

    def save(self, filepath, title=""):
        _ = self.plot(title=title, axis=None, show=False)
        plt.savefig(filepath)

    def plot_as_warpgrid(self, title="", interval=1, show_axis=False, invert_yaxis=True, axis=None):
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
        assert self.dim == 2

        fig_created = False
        if not axis:
            fig_created = True
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

        if fig_created:
            return fig
        return axis

    @property
    def norm(self):
        return np.linalg.norm(self.to_numpy())

    def copy(self):
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

    def to_numpy(self, shape=None):
        if shape:
            return self._data.reshape(shape)
        return self._data


class TimeDependentVectorField:
    def __init__(self, spatial_shape=(), time_steps=1, data=None):
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
                    self._data.append(elem)
                else:
                    self._data.append(VectorField(self.spatial_shape, data=elem))

    def animate(self, title="", interval=1, show_axis=False):
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        if show_axis is False:
            axis.set_axis_off()

        axis.set_aspect('equal')
        axis.set_title(title)

        def animate(i):
            axis.clear()
            self[i].plot(axis=axis)

        ani = animation.FuncAnimation(fig, animate, frames=self.time_steps, interval=100)
        return ani

    @property
    def norm(self):
        return np.linalg.norm(self.to_numpy())

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

    def to_numpy(self, shape=None):
        result = np.array([a.to_numpy() for a in self._data])
        assert result.shape == self.full_shape
        if shape:
            return result.reshape(shape)
        return result

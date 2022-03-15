import numpy as np
from numbers import Number
from copy import deepcopy
import matplotlib.pyplot as plt

from geodesic_shooting.utils.grad import finite_difference


class ScalarFunction:
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
        if data is None:
            assert spatial_shape != ()
        else:
            if spatial_shape != ():
                assert spatial_shape == data.shape
            else:
                spatial_shape = data.shape

        self.spatial_shape = spatial_shape
        self.dim = len(self.spatial_shape)
        self.full_shape = self.spatial_shape

        if data is None:
            data = np.zeros(self.full_shape)
        self._data = data
        assert self._data.shape == self.full_shape

    @property
    def grad(self):
        """Computes the (discrete, approximate) gradient using finite differences.

        Returns
        -------
        `VectorField` representing a finite difference approximation of the gradient.
        """
        return finite_difference(self)

    def to_numpy(self, shape=None):
        """Returns the `ScalarFunction` represented as a numpy-array.

        Parameters
        ----------
        shape
            If not `None`, the numpy-array is reshaped according to `shape`.

        Returns
        -------
        Numpy-array containing the entries of the `ScalarFunction`.
        """
        if shape:
            return self._data.reshape(shape)
        return self._data

    def plot(self, title="", colorbar=True, axis=None):
        """Plots the `ScalarFunction` using `matplotlib`.

        Parameters
        ----------
        title
            The title of the plot.
        colorbar
            Determines whether or not to show a colorbar (only used if `axis=None`).
        axis
            If not `None`, the function is plotted on the provided axis.

        Returns
        -------
        If `axis` is None, the created figure is returned, otherwise a pair containing
        the axis and the return value of `axis.plot` respectively `axis.imshow` (can be
        used to create colorbars) is returned.
        """
        assert self.dim in (1, 2)

        if not axis:
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1)

        if self.dim == 1:
            vals = axis.plot(self.to_numpy())
        else:
            vals = axis.imshow(self.to_numpy().transpose(), origin='lower')

        axis.set_title(title)

        if not axis:
            if colorbar:
                fig.colorbar(vals, ax=axis)
            return fig
        return axis, vals

    def save(self, filepath, title=""):
        """Saves the plot of the `ScalarFunction` produced by the `plot`-function.

        Parameters
        ----------
        filepath
            Path to save the plot to.
        title
            Title of the plot.
        """
        _ = self.plot(title=title, axis=None)
        plt.savefig(filepath)

    def get_norm(self, order=None):
        """Computes the norm of the `ScalarFunction`.

        Remark: If `order=None` and `self.dim >= 2`, the 2-norm of `self.to_numpy().ravel()`
        is returned, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.

        Parameters
        ----------
        order
            Order of the norm,
            see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.

        Returns
        -------
        The norm of the `ScalarFunction`.
        """
        return np.linalg.norm(self.to_numpy(), ord=order)

    norm = property(get_norm)

    @property
    def size(self):
        """Returns the size of the `ScalarFunction`, i.e. the number of entries of the numpy-array.

        Returns
        -------
        The size of the corresponding numpy-array.
        """
        return self._data.size

    def copy(self):
        """Returns a deepcopy of the `ScalarFunction`.

        Returns
        -------
        Deepcopy of the whole `ScalarFunction`.
        """
        return deepcopy(self)

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

    def __eq__(self, other):
        if isinstance(other, ScalarFunction) and (other.to_numpy() == self.to_numpy()).all():
            return True
        return False

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, val):
        if isinstance(val, ScalarFunction):
            self._data[index] = val.to_numpy()
        else:
            self._data[index] = val
        assert self._data.shape == self.full_shape

    def __str__(self):
        return str(self.to_numpy())

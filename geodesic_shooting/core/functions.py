import numpy as np
from numbers import Number
from copy import deepcopy
import matplotlib.pyplot as plt
import imageio
from PIL import Image

from geodesic_shooting.utils.grad import finite_difference


class ScalarFunction:
    def __init__(self, spatial_shape=(), data=None):
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
        return finite_difference(self)

    def plot(self, title="", colorbar=True, axis=None, show=True):
        assert self.dim in (1, 2)

        fig_created = False
        if not axis:
            fig_created = True
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1)

        if self.dim == 1:
            vals = axis.plot(self.to_numpy())
        else:
            vals = axis.imshow(self.to_numpy().transpose(), origin='lower')

        axis.set_title(title)

        if fig_created:
            if colorbar:
                fig.colorbar(vals, ax=axis)
            if show:
                plt.show()
            return fig
        return axis, vals

    def save(self, filepath, title=""):
        _ = self.plot(title=title, axis=None, show=False)
        plt.savefig(filepath)

    @property
    def norm(self):
        return np.linalg.norm(self.to_numpy())

    @property
    def size(self):
        return self._data.size

    def copy(self):
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

    def to_numpy(self, shape=None):
        if shape:
            return self._data.reshape(shape)
        return self._data

import numpy as np
from numbers import Number
import matplotlib.pyplot as plt

from geodesic_shooting.core.base import BaseFunction
from geodesic_shooting.utils import sampler


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

    def push_forward(self, flow, sampler_options={'order': 1, 'mode': 'edge'}):
        """Pushes forward the `ScalarFunction` along a flow.

        Parameters
        ----------
        flow
            `VectorField` containing the flow according to which to push the input forward.
        sampler_options
            Additional options passed to the `warp`-function, see
            https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp.

        Returns
        -------
        `ScalarFunction` of the forward-pushed function.
        """
        return sampler.sample(self, flow, sampler_options=sampler_options)

    def plot(self, title="", colorbar=True, axis=None, extent=(0., 1., 0., 1.)):
        """Plots the `ScalarFunction` using `matplotlib`.

        Parameters
        ----------
        title
            The title of the plot.
        colorbar
            Determines whether or not to show a colorbar (only used if `axis=None`).
        axis
            If not `None`, the function is plotted on the provided axis.
        extent
            Determines the left, right, bottom, and top coordinates of the plot.
            Only used in the 2-dimensional case.

        Returns
        -------
        If `axis` is None, the created figure is returned, otherwise a pair containing
        the axis and the return value of `axis.plot` respectively `axis.imshow` (can be
        used to create colorbars) is returned.
        """
        assert self.dim in (1, 2)

        created_figure = False
        if not axis:
            created_figure = True
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1)

        if self.dim == 1:
            vals = axis.plot(self.to_numpy())
        else:  # self.dim == 2
            vals = axis.imshow(self.to_numpy().transpose(), origin='lower', extent=extent)

        axis.set_title(title)

        if created_figure:
            if colorbar:
                fig.colorbar(vals, ax=axis)
            return fig, axis, vals
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
        try:
            fig, _, _ = self.plot(title=title, axis=None)
            fig.savefig(filepath)
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

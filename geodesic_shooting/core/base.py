import numpy as np
from copy import deepcopy

from geodesic_shooting.utils.helper_functions import tuple_product
import geodesic_shooting.utils.grad as grad
from geodesic_shooting.utils import sampler


class BaseFunction:
    """Class that represents a vector-valued function, i.e. a vector field."""
    def __init__(self, spatial_shape=(), data=None):
        """Constructor.

        Parameters
        ----------
        spatial_shape
            Spatial shape of the underlying domain, i.e. number of components
            in each coordinate direction.
        data
            numpy-array containing the values of the `BaseFunction`. If `None`,
            one has to provide `spatial_shape` and a numpy-array containing zeros
            with shape `(*spatial_shape, dim)` is created as data, where `dim` is
            the dimension of the underlying domain (given as `len(spatial_shape)`).
            If not `None`, either `spatial_shape` is the empty tuple or the first
            components of the shape of `data` (without the last component)
            fit to the provided `spatial_shape`.
        """
        if isinstance(data, BaseFunction):
            data = data._data

        self.spatial_shape = self._compute_spatial_shape(spatial_shape, data)
        self.dim = len(self.spatial_shape)
        self.full_shape = self._compute_full_shape()

        if data is None:
            assert spatial_shape != ()
            data = np.zeros(self.full_shape)
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

    def push_forward(self, flow, sampler_options={'order': 1, 'mode': 'edge'}):
        """Pushes forward the `BaseFunction` along a flow, i.e. compose with the `Diffeomorphism`.

        Parameters
        ----------
        flow
            `Diffeomorphism` containing the flow according to which to push the input forward.
        sampler_options
            Additional options passed to the `warp`-function, see
            https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp.

        Returns
        -------
        The forward-pushed `BaseFunction`.
        """
        return sampler.sample(self, flow, sampler_options=sampler_options)

    def to_numpy(self, shape=None):
        """Returns the `BaseFunction` represented as a numpy-array.

        Parameters
        ----------
        shape
            If not `None`, the numpy-array is reshaped according to `shape`.

        Returns
        -------
        Numpy-array containing the entries of the `BaseFunction`.
        """
        if shape:
            return self._data.reshape(shape)
        return self._data

    def flatten(self):
        """Returns the `BaseFunction` represented as a flattened numpy-array.

        Returns
        -------
        Flattened numpy-array containing the entries of the `BaseFunction`.
        """
        return self.to_numpy().flatten()

    def get_norm(self, product_operator=None, order=None, restriction=np.s_[...]):
        """Computes the norm of the `BaseFunction`.

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
        The norm of the `BaseFunction`.
        """
        vol = 1. / tuple_product(self.spatial_shape)
        if product_operator:
            apply_product_operator = product_operator(self).to_numpy()[restriction].flatten()
            return np.sqrt(apply_product_operator.dot(self.to_numpy()[restriction].flatten())) * np.sqrt(vol)
        else:
            if order is None:
                order = 2
            return np.linalg.norm(self.to_numpy()[restriction].flatten(), ord=order) * np.power(vol, 1./order)

    norm = property(get_norm)

    @property
    def size(self):
        """Returns the size of the `BaseFunction`, i.e. the number of entries of the numpy-array.

        Returns
        -------
        The size of the corresponding numpy-array.
        """
        return self._data.size

    def copy(self):
        """Returns a deepcopy of the `BaseFunction`.

        Returns
        -------
        Deepcopy of the whole `BaseFunction`.
        """
        return deepcopy(self)

    def __eq__(self, other):
        if isinstance(other, BaseFunction) and (other.to_numpy() == self.to_numpy()).all():
            return True
        return False

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, val):
        if isinstance(val, BaseFunction) or isinstance(val, BaseFunction):
            self._data[index] = val._data
        else:
            self._data[index] = val
        assert self._data.shape == self.full_shape

    def __str__(self):
        return str(self.to_numpy())


class BaseTimeDependentFunction:
    """Class that represents a time-dependent vector field."""
    def __init__(self, spatial_shape=(), time_steps=1, data=None, time_independent_type=None):
        """Constructor.

        Parameters
        ----------
        spatial_shape
            Spatial shape of the underlying domain, i.e. number of components
            in each coordinate direction.
        time_steps
            Number of time steps represented in this vector field.
        data
            numpy-array containing the values of the `BaseTimeDependentFunction`.
            If `None`, one has to provide `spatial_shape` and `time_steps`, and a
            list containing for each time step a `BaseFunction` of zeros with shape
            `(*spatial_shape, dim)` is created as data, where `dim` is the dimension of
            the underlying domain (given as `len(spatial_shape)`).
            If not `None`, either a numpy-array or a `BaseTimeDependentFunction` or a list
            of `BaseFunction`s has to be provided.
        """
        assert isinstance(time_steps, int) and time_steps > 0

        if isinstance(data, list) and all(isinstance(d, BaseFunction) for d in data):
            assert len(data) > 0
            spatial_shape = data[0].spatial_shape
            assert all(d.spatial_shape == spatial_shape for d in data)
            time_steps = len(data)

        self.spatial_shape = spatial_shape
        self.dim = len(self.spatial_shape)
        self.time_steps = time_steps
        self.full_shape = self._compute_full_shape()

        self._data = []
        if data is None:
            assert time_independent_type is not None
            for _ in range(self.time_steps):
                self._data.append(time_independent_type(self.spatial_shape))
        else:
            assert ((isinstance(data, np.ndarray) and data.shape == self.full_shape)
                    or (isinstance(data, BaseTimeDependentFunction) and data.full_shape == self.full_shape)
                    or (isinstance(data, list) and all(isinstance(d, BaseFunction) for d in data)))
            for elem in data:
                if isinstance(elem, BaseFunction):
                    self._data.append(elem.copy())
                else:
                    assert time_independent_type is not None
                    self._data.append(time_independent_type(self.spatial_shape, data=elem))

    def _compute_full_shape(self):
        return (self.time_steps, *self.spatial_shape, self.dim)

    def to_numpy(self, shape=None):
        """Returns the `BaseTimeDependentFunction` represented as a numpy-array.

        Parameters
        ----------
        shape
            If not `None`, the numpy-array is reshaped according to `shape`.

        Returns
        -------
        Numpy-array containing the entries of the `BaseTimeDependentFunction`.
        """
        result = np.array([a.to_numpy() for a in self._data])
        assert result.shape == self.full_shape
        if shape:
            return result.reshape(shape)
        return result

    def get_norm(self, order=None, restriction=np.s_[...]):
        """Computes the norm of the `BaseTimeDependentFunction`.

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
        The norm of the `BaseTimeDependentFunction`.
        """
        # TODO: Check this!!!
        return np.linalg.norm(self.to_numpy()[restriction].flatten(), ord=order)

    norm = property(get_norm)

    def __setitem__(self, index, val):
        assert isinstance(val, BaseFunction) and val.full_shape == (*self.spatial_shape, self.dim)
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

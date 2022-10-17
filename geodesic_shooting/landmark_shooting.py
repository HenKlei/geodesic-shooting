# -*- coding: utf-8 -*-
import time
import numpy as np

import scipy.optimize as optimize

from geodesic_shooting.core import VectorField, TimeDependentVectorField
from geodesic_shooting.utils.kernels import GaussianKernel
from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils import sampler
from geodesic_shooting.utils import grid
from geodesic_shooting.utils.time_integration import RK4


class LandmarkShooting:
    """Class that implements geodesic shooting for landmark matching.

    Based on:
    Geodesic Shooting and Diffeomorphic Matching Via Textured Meshes.
    Allassonnière, Trouvé, Younes, 2005
    """
    def __init__(self, kernel=GaussianKernel, kwargs_kernel={}, dim=2, num_landmarks=1,
                 time_integrator=RK4, time_steps=30, sampler_options={'order': 1, 'mode': 'edge'},
                 log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        kernel
            Kernel to use for extending the velocity fields to the whole domain.
        kwargs_kernel
            Additional arguments passed to the constructor of the kernel.
        dim
            Dimension of the landmarks (set automatically when calling `register`).
        num_landmarks
            Number of landmarks to register (set automatically when calling `register`).
        time_integrator
            Method to use for time integration.
        time_steps
            Number of time steps performed during forward and backward integration.
        sampler_options
            Additional options to pass to the sampler.
        log_level
            Verbosity of the logger.
        """
        self.time_integrator = time_integrator
        self.time_steps = time_steps
        self.dt = 1. / self.time_steps

        self.dim = dim
        self.size = dim * num_landmarks

        self.kernel = kernel(**kwargs_kernel)

        self.sampler_options = sampler_options

        self.logger = getLogger('landmark_shooting', level=log_level)

    def __str__(self):
        return (f"{self.__class__.__name__}:\n"
                f"\tKernel:\n{self.kernel}\n"
                f"\tTime integrator: {self.time_integrator.__name__}\n"
                f"\tTime steps: {self.time_steps}\n"
                f"\tSampler options: {self.sampler_options}")

    def register(self, input_landmarks, target_landmarks, landmarks_labeled=True,
                 kernel_dist=GaussianKernel, kwargs_kernel_dist={},
                 sigma=1., optimization_method='L-BFGS-B', optimizer_options={'disp': True},
                 initial_momenta=None, return_all=False):
        """Performs actual registration according to geodesic shooting algorithm for landmarks using
           a Hamiltonian setting.

        Parameters
        ----------
        input_landmarks
            Initial positions of the landmarks.
        target_landmarks
            Target positions of the landmarks.
        landmarks_labeled
            If `True`, the input and target landmarks are assumed to correspond to each other.
            In that case, the l2-mismatch is used as matching function.
            If `False`, all landmarks are treated as dirac measures with weight 1.
            In that case, `kernel_dist` is used as kernel to compute the matching function.
        kernel_dist
            Kernel to use for the matching function.
            Only required if `landmarks_labeled` is `False`.
        kwargs_kernel_dist
            Additional arguments passed to the constructor of the kernel.
            Only required if `landmarks_labeled` is `False`.
        sigma
            Weight for the similarity measurement (L2 difference of the target and the registered
            landmarks); the smaller sigma, the larger the influence of the L2 loss.
        optimization_method
            Optimizer from `scipy`, see `method` under
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        optimizer_options
            Additional options passed to the `scipy.optimize.minimize`-function, see `options` under
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        initial_momenta
            Used as initial guess for the initial momenta (will agree with the direction pointing
            from the input landmarks to the target landmarks if None is passed).
        return_all
            Determines whether or not to return all information or only the initial momenta
            that led to the best registration result.

        Returns
        -------
        Either the best initial momenta (if return_all is False) or a dictionary consisting of the
        transformed/registered landmarks, the initial momenta, the energies and the time evolutions
        of momenta and positions (if return_all is True).
        """
        assert input_landmarks.ndim == 2
        if landmarks_labeled:
            assert input_landmarks.shape == target_landmarks.shape
        self.dim = input_landmarks.shape[1]
        self.size = input_landmarks.shape[0] * input_landmarks.shape[1]

        # define initial momenta
        if initial_momenta is None:
            if landmarks_labeled:
                initial_momenta = (target_landmarks - input_landmarks)
            else:
                initial_momenta = np.zeros(input_landmarks.shape)
        else:
            initial_momenta = np.reshape(initial_momenta, input_landmarks.shape)
        assert initial_momenta.shape == input_landmarks.shape

        initial_momenta = initial_momenta.flatten()
        initial_positions = input_landmarks.flatten()

        if landmarks_labeled:
            def compute_matching_function(positions):
                return np.linalg.norm(positions - target_landmarks.flatten())**2

            def compute_gradient_matching_function(positions):
                return 2. * (positions - target_landmarks.flatten()) / sigma**2
        else:
            kernel_dist = kernel_dist(**kwargs_kernel_dist, scalar=True)

            def compute_matching_function(positions):
                reshaped_positions = positions.reshape(input_landmarks.shape)
                dist = 0.
                for p in reshaped_positions:
                    for q in reshaped_positions:
                        dist += kernel_dist(p, q)
                    for t in target_landmarks:
                        dist -= 2. * kernel_dist(p, t)
                for t in target_landmarks:
                    for s in target_landmarks:
                        dist += kernel_dist(t, s)
                return dist

            def compute_gradient_matching_function(positions):
                grad = np.zeros(positions.shape)
                reshaped_positions = positions.reshape(input_landmarks.shape)
                for i, p in enumerate(reshaped_positions):
                    for q in reshaped_positions:
                        for j in range(self.dim):
                            grad[i*self.dim+j] += kernel_dist.derivative_1(p, q, j)
                            grad[i*self.dim+j] += kernel_dist.derivative_2(q, p, j)
                    for t in target_landmarks:
                        for j in range(self.dim):
                            grad[i*self.dim+j] -= 2. * kernel_dist.derivative_1(p, t, j)
                return grad

        opt = {'input_landmarks': input_landmarks, 'target_landmarks': target_landmarks}

        start_time = time.perf_counter()

        def save_current_state(x):
            opt['x'] = x

        def energy_and_gradient(initial_momenta):
            momenta_time_dependent, positions_time_dependent = self.integrate_forward_Hamiltonian(
                initial_momenta, initial_positions)
            positions = positions_time_dependent[-1]

            energy_regularizer = self.compute_Hamiltonian(initial_momenta, initial_positions)
            energy_l2_unscaled = compute_matching_function(positions)
            energy_l2 = 1. / sigma**2 * energy_l2_unscaled
            energy = energy_regularizer + energy_l2

            d_positions_1, _ = self.integrate_forward_variational_Hamiltonian(momenta_time_dependent,
                                                                              positions_time_dependent)
            positions = positions_time_dependent[-1]
            grad_g = compute_gradient_matching_function(positions)
            grad = self.K(initial_positions) @ initial_momenta + d_positions_1.T @ grad_g / sigma**2

            return energy, grad.flatten()

        # use scipy optimizer for minimizing energy function
        with self.logger.block("Perform landmark matching via geodesic shooting ..."):
            res = optimize.minimize(energy_and_gradient, initial_momenta.flatten(),
                                    method=optimization_method, jac=True, options=optimizer_options,
                                    callback=save_current_state)

        opt['initial_momenta'] = res['x'].reshape(input_landmarks.shape)
        momenta_time_dependent, positions_time_dependent = self.integrate_forward_Hamiltonian(res['x'],
                                                                                              initial_positions)
        opt['registered_landmarks'] = positions_time_dependent[-1].reshape(input_landmarks.shape)
        opt['time_evolution_momenta'] = momenta_time_dependent
        opt['time_evolution_positions'] = positions_time_dependent

        elapsed_time = int(time.perf_counter() - start_time)

        opt['iterations'] = res['nit']
        opt['time'] = elapsed_time
        opt['reason_registration_ended'] = res['message']

        if return_all:
            return opt
        return opt['initial_momenta']

    def compute_Hamiltonian(self, momenta, positions):
        """Computes the value of the Hamiltonian given positions and momenta.

        Parameters
        ----------
        momenta
            Array containing the (time-independent) momenta of the landmarks.
        positions
            Array containing the (time-independent) positions of the landmarks.

        Returns
        -------
        Value of the Hamiltonian.
        """
        assert momenta.shape == (self.size,)
        assert positions.shape == (self.size,)
        return 0.5 * momenta.T @ self.K(positions) @ momenta

    def integrate_forward_Hamiltonian(self, initial_momenta, initial_positions):
        """Performs forward integration of Hamiltonian equations to obtain time-dependent momenta
           and positions.

        Parameters
        ----------
        initial_momenta
            Array containing the initial momenta of the landmarks.
        initial_positions
            Array containing the initial positions of the landmarks.

        Returns
        -------
        Time-dependent momenta and positions.
        """
        assert initial_momenta.shape == (self.size,)
        assert initial_positions.shape == (self.size,)

        momenta = np.zeros((self.time_steps, self.size))
        positions = np.zeros((self.time_steps, self.size))
        momenta[0] = initial_momenta
        positions[0] = initial_positions

        def rhs_momenta_function(momentum, position):
            return - 0.5 * (momentum.T @ self.DK(position) @ momentum)

        ti_momenta = self.time_integrator(rhs_momenta_function, self.dt)

        def rhs_position_function(position, momentum):
            return self.K(position) @ momentum

        ti_position = self.time_integrator(rhs_position_function, self.dt)

        for t in range(self.time_steps-1):
            momenta[t+1] = ti_momenta.step(momenta[t], additional_args={'position': positions[t]})
            positions[t+1] = ti_position.step(positions[t], additional_args={'momentum': momenta[t]})

        return momenta, positions

    def K(self, positions):
        """Computes matrix that contains (dim x dim)-blocks derived from the kernel.

        Parameters
        ----------
        positions
            Array containing the positions of the landmarks.

        Returns
        -------
        Matrix of shape (size x size).
        """
        assert positions.shape == (self.size, )

        mat = []
        pos = positions.reshape((self.size // self.dim, self.dim))

        for i in range(self.size // self.dim):
            mat_row = []
            for j in range(self.size // self.dim):
                mat_row.append(self.kernel(pos[i], pos[j]))
            mat.append(mat_row)

        block_mat = np.block(mat)
        assert block_mat.shape == (self.size, self.size)

        return block_mat

    def DK(self, positions):
        """Computes derivative of the matrix K as a third order tensor.

        Parameters
        ----------
        positions
            Array containing the positions of the landmarks.

        Returns
        -------
        Tensor of shape (size x size x size).
        """
        mat = np.zeros((self.size, self.size, self.size))

        for i in range(self.size):
            modi = i % self.dim
            for j in range(self.size):
                modj = j % self.dim
                for k in range(self.size):
                    if i == k:
                        mat[i][j][k] += self.kernel.derivative_1(positions[i-modi:i+self.dim-modi],
                                                                 positions[j-modj:j+self.dim-modj],
                                                                 modi)
                    if j == k:
                        mat[i][j][k] += self.kernel.derivative_2(positions[i-modi:i+self.dim-modi],
                                                                 positions[j-modj:j+self.dim-modj],
                                                                 modj)

        return mat

    def integrate_forward_variational_Hamiltonian(self, momenta, positions):
        """Performs forward integration of Hamiltonian equations for derivatives of momenta
           and positions with respect to initial momenta and positions.

        Parameters
        ----------
        momenta
            Array containing the time-dependent momenta of the landmarks.
        positions
            Array containing the time-dependent positions of the landmarks.

        Returns
        -------
        Derivative of the momenta at final time t=1.
        """
        assert positions.shape == (self.time_steps, self.size)
        assert momenta.shape == (self.time_steps, self.size)

        d_positions = np.zeros((self.time_steps, self.size, self.size))
        d_momenta = np.zeros((self.time_steps, self.size, self.size))
        d_momenta[0] = np.eye(self.size)

        def rhs_d_positions_function(d_position, position, d_momentum, momentum):
            return self.DK(position) @ d_position @ momentum + self.K(position) @ d_momentum

        ti_d_positions = self.time_integrator(rhs_d_positions_function, self.dt)

        def rhs_d_momenta_function(d_momentum, position):
            return - (d_momentum @ self.DK(position) @ position + position @ self.DK(position) @ d_momentum)

        ti_d_momenta = self.time_integrator(rhs_d_momenta_function, self.dt)

        for t in range(self.time_steps-1):
            d_positions[t+1] = ti_d_positions.step(d_positions[t], additional_args={'position': positions[t],
                                                                                    'd_momentum': d_momenta[t],
                                                                                    'momentum': momenta[t]})
            d_momenta[t+1] = ti_d_momenta.step(d_momenta[t], additional_args={'position': positions[t]})

        return d_positions[-1], d_momenta[-1]

    def get_vector_field(self, momenta, positions,
                         mins=np.array([0., 0.]), maxs=np.array([1., 1.]),
                         spatial_shape=(100, 100)):
        """Evaluates vector field given by positions and momenta at grid points.

        Parameters
        ----------
        momenta
            Array containing the momenta of the landmarks.
        positions
            Array containing the positions of the landmarks.
        mins
            Array containing the lower bounds of the coordinates.
        maxs
            Array containing the upper bounds of the coordinates.
        spatial_shape
            Tuple containing the spatial shape of the grid the diffeomorphism is defined on.

        Returns
        -------
        `VectorField` with the evaluations at the grid points of the vector field defined by
        momenta and positions.
        """
        vector_field = VectorField(spatial_shape)

        vf_func = construct_vector_field(momenta.reshape((-1, self.dim)),
                                         positions.reshape((-1, self.dim)),
                                         kernel=self.kernel)

        for pos in np.ndindex(spatial_shape):
            spatial_pos = mins + (maxs - mins) / np.array(spatial_shape) * np.array(pos)
            vector_field[pos] = vf_func(spatial_pos) * np.array(spatial_shape)

        return vector_field

    def compute_time_evolution_of_diffeomorphisms(self, initial_momenta, initial_positions,
                                                  mins=np.array([0., 0.]), maxs=np.array([1., 1.]),
                                                  spatial_shape=(100, 100)):
        """Performs forward integration of diffeomorphism on given grid using the given
           initial momenta and positions.

        Parameters
        ----------
        initial_momenta
            Array containing the initial momenta of the landmarks.
        initial_positions
            Array containing the initial positions of the landmarks.
        mins
            Array containing the lower bounds of the coordinates.
        maxs
            Array containing the upper bounds of the coordinates.
        spatial_shape
            Tuple containing the spatial shape of the grid the diffeomorphism is defined on.

        Returns
        -------
        `VectorField` containing the diffeomorphism at the different time instances.
        """
        assert mins.ndim == 1 and mins.shape[0] == len(spatial_shape)
        assert maxs.ndim == 1 and maxs.shape[0] == len(spatial_shape)
        assert np.all(mins < maxs)
        assert initial_momenta.shape == initial_positions.shape

        momenta, positions = self.integrate_forward_Hamiltonian(initial_momenta.flatten(), initial_positions.flatten())
        vector_fields = TimeDependentVectorField(spatial_shape, self.time_steps)

        for t, (m, p) in enumerate(zip(momenta, positions)):
            vector_fields[t] = self.get_vector_field(m, p, mins, maxs, spatial_shape)

        flow = self.integrate_forward_flow(vector_fields)

        return flow

    def integrate_forward_flow(self, vector_fields):
        """Computes forward integration according to given vector fields.

        Parameters
        ----------
        vector_fields
            `TimeDependentVectorField` containing the sequence of vector fields to integrate
            in time.

        Returns
        -------
        `VectorField` containing the flow at the final time.
        """
        assert isinstance(vector_fields, TimeDependentVectorField)
        assert vector_fields.time_steps == self.time_steps
        spatial_shape = vector_fields[0].spatial_shape
        # make identity grid
        identity_grid = grid.coordinate_grid(spatial_shape)

        # initial flow is the identity mapping
        flow = identity_grid.copy()

        def rhs_function(x, v):
            return - sampler.sample(v, x, sampler_options=self.sampler_options)

        ti = self.time_integrator(rhs_function, self.dt)

        # perform forward integration
        for v in vector_fields:
            flow = ti.step(flow, additional_args={'v': v})

        return flow


def construct_vector_field(momenta, positions, kernel=GaussianKernel()):
    """Computes the vector field corresponding to the given positions and momenta.

    Parameters
    ----------
    momenta
        Array containing the momenta of the landmarks.
    positions
        Array containing the positions of the landmarks.

    Returns
    -------
    Function that can be evaluated at any point of the space.
    """
    assert positions.ndim == 2
    assert positions.shape == momenta.shape

    def vector_field(x):
        result = np.zeros(positions.shape[1])
        for q, p in zip(positions, momenta):
            result += kernel(x, q) @ p
        return result

    return vector_field

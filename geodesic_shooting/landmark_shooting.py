# -*- coding: utf-8 -*-
import time
import numpy as np

import scipy.optimize as optimize

from geodesic_shooting.core import VectorField, TimeDependentVectorField
from geodesic_shooting.utils.kernels import GaussianKernel
from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.optimizers import gradient_descent
from geodesic_shooting.utils.time_integration import RK4


class LandmarkShooting:
    """Class that implements geodesic shooting for landmark matching.

    Based on:
    Geodesic Shooting and Diffeomorphic Matching Via Textured Meshes.
    Allassonnière, Trouvé, Younes, 2005
    """
    def __init__(self, kernel=GaussianKernel, kwargs_kernel={}, dim=2, num_landmarks=1,
                 time_integrator=RK4, time_steps=100, sampler_options={'order': 1, 'mode': 'edge'},
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
        self.dt = 1. / (self.time_steps - 1.)

        self.dim = dim
        self.num_landmarks = num_landmarks
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
                 sigma=1., optimization_method='L-BFGS-B', optimizer_options={'disp': True, 'maxiter': 1000},
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
            Determines whether to return all information or only the initial momenta
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
        self.num_landmarks = input_landmarks.shape[0]
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

        initial_momenta = initial_momenta
        initial_positions = input_landmarks

        if landmarks_labeled:
            def compute_matching_function(positions):
                return np.linalg.norm(positions.flatten() - target_landmarks.flatten())**2

            def compute_gradient_matching_function(positions):
                return 2. * (positions - target_landmarks.flatten())
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

        def energy_and_gradient(initial_momenta, compute_grad=True, return_all_energies=False):
            momenta_time_dependent, positions_time_dependent = self.integrate_forward_Hamiltonian(
                initial_momenta, initial_positions)
            positions = positions_time_dependent[-1]

            energy_regularizer = self.compute_Hamiltonian(initial_momenta, initial_positions)
            energy_intensity_unscaled = compute_matching_function(positions)
            energy_intensity = 1. / sigma**2 * energy_intensity_unscaled
            energy = energy_regularizer + energy_intensity

            if compute_grad:
                d_positions_1, _ = self.integrate_forward_variational_Hamiltonian(momenta_time_dependent,
                                                                                  positions_time_dependent)

                grad = np.zeros((self.num_landmarks, self.dim))
                assert initial_momenta.shape == (self.num_landmarks, self.dim)
                assert initial_positions.shape == (self.num_landmarks, self.dim)
                for c, (pc, qc) in enumerate(zip(initial_momenta, initial_positions)):
                    for j in range(self.dim):
                        for a, (pa, qa) in enumerate(zip(initial_momenta, initial_positions)):
                            for i in range(self.dim):
                                grad[c, j] += self.kernel(qa, qc)[i, j] * pa[i] * pc[j]

                assert positions.shape == (self.num_landmarks, self.dim)
                assert target_landmarks.shape == (self.num_landmarks, self.dim)
                for c in range(self.num_landmarks):
                    for j in range(self.dim):
                        for a, (qa1, target_qa1) in enumerate(zip(positions, target_landmarks)):
                            for i in range(self.dim):
                                grad[c, j] += 2. * (qa1[i] - target_qa1[i]) * d_positions_1[a, i, c, j] / sigma**2

                if return_all_energies:
                    return energy, energy_regularizer, energy_intensity_unscaled, energy_intensity, grad
                return energy, grad
            else:
                if return_all_energies:
                    return energy, energy_regularizer, energy_intensity_unscaled, energy_intensity
                else:
                    return energy

        if optimization_method == 'newton' and landmarks_labeled:
            # use Newton's method for minimizing energy function
            def newton(x0, update_norm_tol=1e-5, rel_func_update_tol=1e-6, maxiter=50, disp=True, callback=None):
                assert update_norm_tol >= 0 and rel_func_update_tol >= 0
                assert isinstance(maxiter, int) and maxiter > 0

                def compute_update_direction(x):
                    momenta_time_dependent, positions_time_dependent = self.integrate_forward_Hamiltonian(
                        x, initial_positions)
                    momenta = momenta_time_dependent[-1]
                    positions = positions_time_dependent[-1]
                    d_positions_1, d_momenta_1 = self.integrate_forward_variational_Hamiltonian(
                        momenta_time_dependent, positions_time_dependent)
                    mat = d_momenta_1 + 2 * np.eye(self.size) @ d_positions_1 / sigma ** 2
                    _, grad = energy_and_gradient(x)
                    update = np.linalg.solve(mat, momenta + (positions - target_landmarks.flatten()) / sigma ** 2)
                    return update

                message = ''
                with self.logger.block('Starting optimization using Newton Algorithm ...'):
                    x = x0.flatten()
                    func_x, _ = energy_and_gradient(x)
                    old_func_x = func_x
                    rel_func_update = rel_func_update_tol + 1
                    update = compute_update_direction(x)
                    norm_update = np.linalg.norm(update)
                    i = 0
                    if disp:
                        self.logger.info(f'iter: {i:5d}\tf= {func_x:.5e}\t|update|= {norm_update:.5e}\t'
                                         f'rel.func.upd.= {rel_func_update:.5e}')
                    try:
                        while True:
                            if callback is not None:
                                callback(np.copy(x))
                            if norm_update <= update_norm_tol:
                                message = 'norm of update below tolerance'
                                break
                            elif rel_func_update <= rel_func_update_tol:
                                message = 'relative function value update below tolerance'
                                break
                            elif i >= maxiter:
                                message = 'maximum number of iterations reached'
                                break

                            update = compute_update_direction(x)
                            x = x - update

                            func_x, _ = energy_and_gradient(x)
                            if not np.isclose(old_func_x, 0.):
                                rel_func_update = abs((func_x - old_func_x) / old_func_x)
                            else:
                                rel_func_update = 0.
                            old_func_x = func_x
                            norm_update = np.linalg.norm(update)
                            i += 1
                            if disp:
                                self.logger.info(f'iter: {i:5d}\tf= {func_x:.5e}\t|update|= {norm_update:.5e}\t'
                                                 f'rel.func.upd.= {rel_func_update:.5e}')
                    except KeyboardInterrupt:
                        message = 'optimization stopped due to keyboard interrupt'
                        self.logger.warning('Optimization interrupted ...')

                self.logger.info('Finished optimization ...')
                result = {'x': x, 'nit': i, 'message': message}
                return result

            res = newton(initial_momenta, callback=save_current_state, **optimizer_options)
        elif optimization_method == 'newton' and not landmarks_labeled:
            raise NotImplementedError
        elif optimization_method == 'GD':
            res = gradient_descent(energy_and_gradient, initial_momenta,
                                   logger=self.logger, callback=save_current_state, **optimizer_options)
        else:
            # use scipy optimizer for minimizing energy function
            with self.logger.block("Perform landmark matching via geodesic shooting ..."):
                res = optimize.minimize(energy_and_gradient, initial_momenta.flatten(),
                                        method=optimization_method, jac=True, options=optimizer_options,
                                        callback=save_current_state)

        opt['initial_momenta'] = res['x']
        momenta_time_dependent, positions_time_dependent = self.integrate_forward_Hamiltonian(res['x'],
                                                                                              initial_positions)
        opt['registered_landmarks'] = positions_time_dependent[-1].reshape(input_landmarks.shape)
        opt['time_evolution_momenta'] = momenta_time_dependent
        opt['time_evolution_positions'] = positions_time_dependent

        elapsed_time = int(time.perf_counter() - start_time)

        opt['iterations'] = res['nit']
        opt['time'] = elapsed_time
        opt['reason_registration_ended'] = res['message']

        energy, energy_regularizer, energy_intensity_unscaled, energy_intensity, grad = energy_and_gradient(
            opt['initial_momenta'], return_all_energies=True)
        opt['energy'] = energy
        opt['energy_regularizer'] = energy_regularizer
        opt['energy_intensity_unscaled'] = energy_intensity_unscaled
        opt['energy_intensity'] = energy_intensity
        opt['grad'] = grad

        if return_all:
            return opt
        return opt['initial_momenta']

    def _momenta_positions_flatten(self, momenta, positions):
        if not momenta.shape == (self.size,):
            momenta = momenta.flatten()
        if not positions.shape == (self.size,):
            positions = positions.flatten()
        assert momenta.shape == (self.size,)
        assert positions.shape == (self.size,)
        return momenta, positions

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
        momenta, positions = self._momenta_positions_flatten(momenta, positions)
        return 0.5 * momenta.T @ self.K(positions) @ momenta

    def _rhs_momenta_function(self, momenta, positions):
        rhs = np.zeros((self.num_landmarks, self.dim))

        for a, (pa, qa) in enumerate(zip(momenta.reshape((self.num_landmarks, self.dim)),
                                         positions.reshape((self.num_landmarks, self.dim)))):
            for b, (pb, qb) in enumerate(zip(momenta.reshape((self.num_landmarks, self.dim)),
                                             positions.reshape((self.num_landmarks, self.dim)))):
                rhs[a] -= np.einsum("ijk,i,j->k", self.kernel.full_derivative_1(qa, qb), pa, pb)

        return rhs.flatten()

    def _rhs_position_function(self, positions, momenta):
        return self.K(positions) @ momenta

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
        momenta = np.zeros((self.time_steps, self.size))
        positions = np.zeros((self.time_steps, self.size))
        momenta[0] = initial_momenta.flatten()
        positions[0] = initial_positions.flatten()

        ti_momenta = self.time_integrator(self._rhs_momenta_function, self.dt)

        ti_position = self.time_integrator(self._rhs_position_function, self.dt)

        for t in range(self.time_steps-1):
            momenta[t+1] = ti_momenta.step(momenta[t], additional_args={'positions': positions[t]})
            positions[t+1] = ti_position.step(positions[t], additional_args={'momenta': momenta[t]})

        return momenta.reshape((self.time_steps, self.num_landmarks, self.dim)), positions.reshape((self.time_steps, self.num_landmarks, self.dim))

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
        return self.kernel.apply_vectorized(positions, positions, self.dim)

    def _rhs_d_positions_function(self, d_position, position, d_momentum, momentum):
        momentum = momentum.reshape((-1, self.dim))
        position = position.reshape((-1, self.dim))

        rhs = np.zeros((self.num_landmarks, self.dim, self.num_landmarks, self.dim))

        for a, (pa, qa, dqa) in enumerate(zip(momentum, position, d_position)):
            assert dqa.shape == (self.dim, self.num_landmarks, self.dim)
            for c in range(self.num_landmarks):
                for i in range(self.dim):
                    for j in range(self.dim):
                        for b, (pb, qb, dpb, dqb) in enumerate(zip(momentum, position, d_momentum, d_position)):
                            for k in range(self.dim):
                                rhs[a, i, c, j] += self.kernel(qa, qb)[i, k] * dpb[k, c, j]
                                for l in range(self.dim):
                                    rhs[a, i, c, j] += self.kernel.derivative_1(qa, qb, l)[i, k] * dqa[l, c, j] * pb[k]
                                    rhs[a, i, c, j] += self.kernel.derivative_2(qa, qb, l)[i, k] * dqb[l, c, j] * pb[k]

        return rhs

    def _rhs_d_momenta_function(self, d_momentum, momentum, position, d_position):
        momentum = momentum.reshape((-1, self.dim))
        position = position.reshape((-1, self.dim))

        rhs = np.zeros((self.num_landmarks, self.dim, self.num_landmarks, self.dim))

        for a, (pa, qa, dpa, dqa) in enumerate(zip(momentum, position, d_momentum, d_position)):
            assert dqa.shape == (self.dim, self.num_landmarks, self.dim)
            for c in range(self.num_landmarks):
                for i in range(self.dim):
                    for j in range(self.dim):
                        for b, (pb, qb, dpb, dqb) in enumerate(zip(momentum, position, d_momentum, d_position)):
                            for j_tilde in range(self.dim):
                                for k in range(self.dim):
                                    rhs[a, i, c, j] -= self.kernel.derivative_1(qa, qb, i)[j_tilde, k] * dpa[j_tilde, c, j] * pb[k]
                                    rhs[a, i, c, j] -= self.kernel.derivative_1(qa, qb, i)[j_tilde, k] * pa[j_tilde] * dpb[k, c, j]
                                    for l in range(self.dim):
                                        rhs[a, i, c, j] -= self.kernel.second_derivative_1_1(qa, qb, i, l)[j_tilde, k] * dqa[l, c, j]
                                        rhs[a, i, c, j] -= self.kernel.second_derivative_1_2(qa, qb, i, l)[j_tilde, k] * dqb[l, c, j]

        return rhs

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
        assert positions.shape == (self.time_steps, self.num_landmarks, self.dim)
        assert momenta.shape == (self.time_steps, self.num_landmarks, self.dim)

        d_positions = np.zeros((self.time_steps, self.num_landmarks, self.dim, self.num_landmarks, self.dim))
        d_momenta = np.zeros((self.time_steps, self.num_landmarks, self.dim, self.num_landmarks, self.dim))

        for k in range(self.num_landmarks):
            d_momenta[0, k, :, k, :] = np.eye(self.dim)

        ti_d_positions = self.time_integrator(self._rhs_d_positions_function, self.dt)

        ti_d_momenta = self.time_integrator(self._rhs_d_momenta_function, self.dt)

        for t in range(self.time_steps-1):
            d_positions[t+1] = ti_d_positions.step(d_positions[t], additional_args={'position': positions[t],
                                                                                    'd_momentum': d_momenta[t],
                                                                                    'momentum': momenta[t]})
            d_momenta[t+1] = ti_d_momenta.step(d_momenta[t], additional_args={'momentum': momenta[t],
                                                                              'position': positions[t],
                                                                              'd_position': d_positions[t]})

        return d_positions[-1], d_momenta[-1]

    def get_vector_field(self, momenta, positions, spatial_shape=(100, 100)):
        """Evaluates vector field given by positions and momenta at grid points.

        Parameters
        ----------
        momenta
            Array containing the momenta of the landmarks.
        positions
            Array containing the positions of the landmarks.
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
            spatial_pos = np.array(pos) / np.array(spatial_shape)
            vector_field[pos] = vf_func(spatial_pos)

        return vector_field

    def compute_diffeomorphism(self, initial_momenta, initial_positions, spatial_shape=(100, 100),
                               get_time_dependent_diffeomorphism=False):
        """Performs forward integration of diffeomorphism on given grid using the given
           initial momenta and positions.

        Parameters
        ----------
        initial_momenta
            Array containing the initial momenta of the landmarks.
        initial_positions
            Array containing the initial positions of the landmarks.
        spatial_shape
            Tuple containing the spatial shape of the grid the diffeomorphism is defined on.
        get_time_dependent_diffeomorphism
            Determines whether to return the `TimeDependentDiffeomorphism` or only
            the final `Diffeomorphism`.

        Returns
        -------
        `VectorField` containing the diffeomorphism at the different time instances.
        """
        assert initial_momenta.shape == initial_positions.shape

        momenta, positions = self.integrate_forward_Hamiltonian(initial_momenta, initial_positions)
        vector_fields = TimeDependentVectorField(spatial_shape, self.time_steps)

        for t, (m, p) in enumerate(zip(momenta, positions)):
            vector_fields[t] = self.get_vector_field(m, p, spatial_shape)

        flow = vector_fields.integrate(sampler_options=self.sampler_options,
                                       get_time_dependent_diffeomorphism=get_time_dependent_diffeomorphism)

        return flow


def construct_vector_field(momenta, positions, kernel=GaussianKernel()):
    """Computes the vector field corresponding to the given positions and momenta.

    Parameters
    ----------
    momenta
        Array containing the momenta of the landmarks.
    positions
        Array containing the positions of the landmarks.
    kernel
        Kernel function to use for constructing the vector field.

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

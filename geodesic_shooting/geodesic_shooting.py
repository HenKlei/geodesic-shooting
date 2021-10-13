import time
import numpy as np

import scipy.optimize as optimize

from geodesic_shooting.utils import sampler, grid
from geodesic_shooting.utils.grad import finite_difference
from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer
from geodesic_shooting.utils.optim import GradientDescentOptimizer, ArmijoLineSearch


class GeodesicShooting:
    """Class that implements large deformation metric mappings via geodesic shooting.

    Based on:
    Geodesic Shooting for Computational Anatomy.
    Miller, Trouvé, Younes, 2006
    """
    def __init__(self, alpha=6., exponent=1., dim=2, shape=(100, 100), log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        alpha
            Parameter for biharmonic regularizer.
        exponent
            Parameter for biharmonic regularizer.
        dim
            Dimension of the input and target images (set automatically when calling `register`).
        shape
            Shape of the input and target images (set automatically when calling `register`).
        log_level
            Verbosity of the logger.
        """
        self.regularizer = BiharmonicRegularizer(alpha, exponent)

        self.time_steps = 30
        self.shape = shape
        self.dim = dim
        assert self.dim == len(self.shape)

        self.logger = getLogger('geodesic_shooting', level=log_level)

    def register(self, input_, target, time_steps=30, sigma=1.,
                 OptimizationAlgorithm=GradientDescentOptimizer, iterations=1000, early_stopping=10,
                 initial_velocity_field=None, LineSearchAlgorithm=ArmijoLineSearch,
                 parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1.,
                                         'max_num_search_steps': 10},
                 energy_threshold=1e-6, gradient_norm_threshold=1e-6,
                 return_all=False):
        """Performs actual registration according to LDDMM algorithm with time-varying velocity
           fields that are chosen via geodesics.

        Parameters
        ----------
        input_
            Input image as array.
        target
            Target image as array.
        time_steps
            Number of discrete time steps to perform.
        iterations
            Number of iterations of the optimizer to perform. The value `None` is also possible
            to not bound the number of iterations.
        sigma
            Weight for the similarity measurement (L2 difference of the target and the registered
            image); the smaller sigma, the larger the influence of the L2 loss.
        early_stopping
            Number of iterations with non-decreasing energy after which to stop registration.
            If `None`, no early stopping is used.
        initial_velocity_field
            Used as initial guess for the initial velocity field (will be 0 if None is passed).
        return_all
            Determines whether or not to return all information or only the initial vector field
            that led to the best registration result.

        Returns
        -------
        Either the best initial vector field (if return_all is False) or a dictionary consisting
        of the registered image, the velocities, the energies, the flows and inverse flows, the
        forward-pushed input and the back-pulled target at all time instances (if return_all is
        True).
        """
        assert isinstance(time_steps, int) and time_steps > 0
        assert iterations is None or (isinstance(iterations, int) and iterations > 0)
        assert sigma > 0
        assert (isinstance(early_stopping, int) and early_stopping > 0) or early_stopping is None
        assert input_.shape == target.shape

        input_ = input_.astype('double')
        target = target.astype('double')

        def compute_energy(image):
            return np.sum((image - target)**2)

        def compute_grad_energy(image_gradient, image):
            return self.regularizer.cauchy_navier_squared_inverse(
                image_gradient * (image - target)[np.newaxis, ...])

        # set up variables
        self.time_steps = time_steps
        self.shape = input_.shape
        self.dim = input_.ndim

        self.energy_threshold = energy_threshold
        self.gradient_norm_threshold = gradient_norm_threshold

        # define vector fields
        if initial_velocity_field is None:
            initial_velocity_field = np.zeros((self.dim, *self.shape), dtype=np.double)
        assert initial_velocity_field.shape == (self.dim, *self.shape)

        velocity_fields = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)

        def set_opt(opt, energy, energy_regularizer, energy_intensity, energy_intensity_unscaled,
                    transformed_input, initial_velocity_field, flow):
            opt['energy'] = energy
            opt['energy_regularizer'] = energy_regularizer
            opt['energy_intensity'] = energy_intensity
            opt['energy_intensity_unscaled'] = energy_intensity_unscaled
            opt['transformed_input'] = transformed_input
            opt['initial_velocity_field'] = initial_velocity_field
            opt['flow'] = flow
            return opt

        opt = set_opt({}, None, None, None, None, input_, initial_velocity_field,
                      self.integrate_forward_flow(velocity_fields))

        k = 0
        reason_registration_ended = 'reached maximum number of iterations'

        start_time = time.perf_counter()

        res = {}

        def energy_and_gradient(v0):
            # integrate initial velocity field forward in time
            velocity_fields = self.integrate_forward_vector_field(v0)

            # compute forward flows according to the velocity fields
            flow = self.integrate_forward_flow(velocity_fields)

            # push-forward input_ image
            forward_pushed_input = self.push_forward(input_, flow)

            # compute gradient of the forward-pushed image
            gradient_forward_pushed_input = self.image_grad(forward_pushed_input)

            # compute gradient of the intensity difference
            gradient_l2_energy = (1 / sigma**2
                                  * compute_grad_energy(gradient_forward_pushed_input,
                                                        forward_pushed_input))

            # compute gradient of the intensity difference with respect to the
            # initial velocity
            gradient_initial_velocity = -self.integrate_backward_adjoint_Jacobi_field(
                gradient_l2_energy, velocity_fields)

            # compute the current energy consisting of intensity difference
            # and regularization
            energy_regularizer = np.linalg.norm(self.regularizer.cauchy_navier(
                initial_velocity_field))
            energy_intensity_unscaled = compute_energy(forward_pushed_input)
            energy_intensity = 1 / sigma**2 * energy_intensity_unscaled
            energy = energy_regularizer + energy_intensity

            return energy, gradient_initial_velocity

        line_search = LineSearchAlgorithm(energy_and_gradient)
        optimizer = OptimizationAlgorithm(line_search, energy_and_gradient)

        reason_registration_ended = 'reached maximum number of iterations'

        with self.logger.block("Perform image matching via geodesic shooting ..."):
            try:
                k = 0
                energy_did_not_decrease = 0
                x = res['x'] = initial_velocity_field
                energy, grad = energy_and_gradient(res['x'])
                min_energy = energy

                while not (iterations is not None and k >= iterations):
                    x, energy, grad, _ = optimizer.step(x, energy, grad, parameters_line_search)
                    self.logger.info(f"iter: {k:3d}, energy: {energy:.4e}")

                    if min_energy >= energy:
                        res['x'] = x.copy()
                        min_energy = energy
                        if min_energy < self.energy_threshold:
                            self.logger.info(f"Energy below threshold of {self.energy_threshold}. "
                                             "Stopping ...")
                            reason_registration_ended = 'reached energy threshold'
                            break
                    else:
                        energy_did_not_decrease += 1

                    norm_gradient = np.linalg.norm(grad.flatten())
                    if norm_gradient < self.gradient_norm_threshold:
                        self.logger.warning(f"Gradient norm is {norm_gradient} "
                                            "and therefore below threshold. Stopping ...")
                        reason_registration_ended = 'reached gradient norm threshold'
                        break
                    if early_stopping is not None and energy_did_not_decrease >= early_stopping:
                        reason_registration_ended = 'early stopping due to non-decreasing energy'
                        break
                    k += 1
            except KeyboardInterrupt:
                self.logger.warning("Aborting registration ...")
                reason_registration_ended = 'manual abort'

            velocity_fields = self.integrate_forward_vector_field(res['x'])

            # compute forward flows according to the velocity fields
            flow = self.integrate_forward_flow(velocity_fields)

            # push-forward input_ image
            transformed_input = self.push_forward(input_, flow)

        res['fun'] = energy_and_gradient(res['x'])[0]
        set_opt(opt, res['fun'], None, None, None, transformed_input,
                res['x'].reshape((self.dim, *self.shape)), flow)

        elapsed_time = int(time.perf_counter() - start_time)

        self.logger.info("Finished registration ...")

        if opt['energy'] is not None:
            self.logger.info(f"Optimal energy: {opt['energy']:4.4f}")
#            self.logger.info("Optimal intensity difference (with scale): "
#                             f"{opt['energy_intensity']:4.4f}")
#            self.logger.info("Optimal intensity difference (without scale): "
#                             f"{opt['energy_intensity_unscaled']:4.4f}")
#            self.logger.info(f"Optimal regularization: {opt['energy_regularizer']:4.4f}")

        if opt['initial_velocity_field'] is not None:
            # compute the length of the path on the manifold;
            # this step only requires the initial velocity due to conservation of momentum
            length = np.linalg.norm(self.regularizer.cauchy_navier(
                         opt['initial_velocity_field']))
        else:
            length = 0.0

        opt['length'] = length
        opt['iterations'] = k
        opt['time'] = elapsed_time
        opt['reason_registration_ended'] = reason_registration_ended

        if return_all:
            return opt
        return initial_velocity_field

    def integrate_forward_flow(self, velocity_fields):
        """Computes forward integration according to given velocity fields.

        Parameters
        ----------
        velocity_fields
            Sequence of velocity fields (i.e. time-depending velocity field).

        Returns
        -------
        Array containing the flow at the final time.
        """
        # make identity grid
        identity_grid = grid.coordinate_grid(self.shape)

        # create flow
        flow = np.zeros((self.dim, *self.shape), dtype=np.double)

        # initial flow is the identity mapping
        flow = identity_grid.astype(np.double)

        for t in range(0, self.time_steps-1):
            alpha = self.forward_alpha(velocity_fields[t])
            flow = sampler.sample(flow, identity_grid - alpha)

        return flow

    def forward_alpha(self, velocity_field):
        """Helper function to estimate the updated positions (forward calculation).

        Parameters
        ----------
        velocity_field
            Array containing the velocity field used for updating the positions.

        Returns
        -------
        Array with the update of the positions.
        """
        # make identity grid
        identity_grid = grid.coordinate_grid(self.shape)

        alpha = np.zeros(velocity_field.shape, dtype=np.double)
        for _ in range(5):
            alpha = sampler.sample(velocity_field, identity_grid - 0.5 * alpha)
        return alpha

    def push_forward(self, image, flow):
        """Pushes forward an image along a flow.

        Parameters
        ----------
        image
            Array to push forward.
        flow
            Array containing the flow along which to push the input forward.

        Returns
        -------
        Array with the forward-pushed image.
        """
        return sampler.sample(image, flow)

    def image_grad(self, image):
        """Computes the gradients of the given image.

        Parameters
        ----------
        image
            Array containing the (forward) pushed image.

        Returns
        -------
        Array with the gradients of the input image.
        """
        return finite_difference(image)

    def integrate_forward_vector_field(self, initial_velocity_field):
        """Performs forward integration of the initial velocity field.

        Parameters
        ----------
        initial_velocity_field
            Array with the initial velocity field to integrate forward.

        Returns
        -------
        Sequence of velocity fields obtained via forward integration of the initial velocity field.
        """
        velocity_fields = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)
        velocity_fields[0] = initial_velocity_field.reshape((self.dim, *self.shape))

        einsum_string = 'kl...,l...->k...'
        einsum_string_transpose = 'lk...,l...->k...'

        for t in range(0, self.time_steps-2):
            momentum_t = self.regularizer.cauchy_navier(velocity_fields[t])
            grad_mt = finite_difference(momentum_t)
            grad_vt = finite_difference(velocity_fields[t])
            div_vt = np.sum(np.array([grad_vt[d, d, ...] for d in range(self.dim)]), axis=0)
            rhs = (np.einsum(einsum_string_transpose, grad_vt, momentum_t)
                   + np.einsum(einsum_string, grad_mt, velocity_fields[t])
                   + momentum_t * div_vt[np.newaxis, ...])
            velocity_fields[t+1] = (velocity_fields[t]
                                    - self.regularizer.cauchy_navier_squared_inverse(rhs)
                                    / self.time_steps)

        return velocity_fields

    def integrate_backward_adjoint_Jacobi_field(self, gradient_l2_energy, velocity_fields):
        """Performs backward integration of the adjoint jacobi field equations.

        Parameters
        ----------
        gradient_l2_energy
            Array containing the gradient of the L2 energy functional.
        velocity_fields
            Sequence of velocity fields (i.e. time-dependent velocity field) to integrate backwards.

        Returns
        -------
        Gradient of the energy with respect to the initial velocity field.
        """
        v_old = gradient_l2_energy
        delta_v_old = np.zeros(v_old.shape, dtype=np.double)
        delta_v = delta_v_old

        einsum_string = 'kl...,l...->k...'
        einsum_string_transpose = 'lk...,l...->k...'

        for t in range(self.time_steps-2, -1, -1):
            grad_velocity_fields = finite_difference(velocity_fields[t])
            div_velocity_fields = np.sum(np.array([grad_velocity_fields[d, d, ...]
                                                   for d in range(self.dim)]), axis=0)
            regularized_v = self.regularizer.cauchy_navier(v_old)
            grad_regularized_v = finite_difference(regularized_v)
            rhs_v = - self.regularizer.cauchy_navier_squared_inverse(
                np.einsum(einsum_string_transpose, grad_velocity_fields, regularized_v)
                + np.einsum(einsum_string, grad_regularized_v, velocity_fields[t])
                + regularized_v * div_velocity_fields[np.newaxis, ...])
            v_old = v_old - rhs_v / self.time_steps

            grad_delta_v = finite_difference(delta_v)
            div_delta_v = np.sum(np.array([grad_delta_v[d, d, ...]
                                           for d in range(self.dim)]), axis=0)
            regularized_velocity_fields = self.regularizer.cauchy_navier(velocity_fields[t])
            grad_regularized_velocity_fields = finite_difference(regularized_velocity_fields)
            rhs_delta_v = (- v_old
                           - (np.einsum(einsum_string, grad_velocity_fields, delta_v)
                              - np.einsum(einsum_string, grad_delta_v, velocity_fields[t]))
                           + self.regularizer.cauchy_navier_squared_inverse(
                               np.einsum(einsum_string_transpose, grad_delta_v,
                                         regularized_velocity_fields)
                               + np.einsum(einsum_string, grad_regularized_velocity_fields, delta_v)
                               + regularized_velocity_fields * div_delta_v[np.newaxis, ...]))
            delta_v = delta_v_old - rhs_delta_v / self.time_steps
            delta_v_old = delta_v

        return delta_v

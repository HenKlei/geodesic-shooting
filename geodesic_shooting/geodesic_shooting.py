import numpy as np

from geodesic_shooting.utils import sampler, grid
from geodesic_shooting.utils.grad import finite_difference
from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


class GeodesicShooting:
    """Class that implements large deformation metric mappings via geodesic shooting.

    Based on:
    Geodesic Shooting for Computational Anatomy.
    Miller, Trouvé, Younes, 2006
    """
    def __init__(self, alpha=6., exponent=1., log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        alpha
            Parameter for biharmonic regularizer.
        exponent
            Parameter for biharmonic regularizer.
        """
        self.regularizer = BiharmonicRegularizer(alpha, exponent)

        self.time_steps = 30
        self.shape = None
        self.dim = None
        self.opt = None
        self.opt_energy = None
        self.opt_energy_regularizer = None
        self.opt_energy_intensity = None
        self.energy_threshold = 1e-3
        self.gradient_norm_threshold = 1e-3

        self.logger = getLogger('geodesic_shooting', level=log_level)

    def register(self, input_, target, time_steps=30, iterations=100, sigma=1, epsilon=0.01,
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
            Number of iterations of the optimizer to perform.
        sigma
            Weight for the similarity measurement (L2 difference of the target and the registered
            image); the smaller sigma, the larger the influence of the L2 loss.
        epsilon
            Learning rate, i.e. step size of the optimizer.
        return_all
            Determines whether or not to return all information or only the initial vector field
            that led to the best registration result.

        Returns
        -------
        Either the best initial vector field (if return_all is False) or a tuple consisting of the
        registered image, the velocities, the energies, the flows and inverse flows, the
        forward-pushed input and the back-pulled target at all time instances (if return_all is
        True).
        """
        assert isinstance(time_steps, int) and time_steps > 0
        assert isinstance(iterations, int) and iterations > 0
        assert sigma > 0
        assert 0 < epsilon < 1
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
        self.opt = (input_, None, [], [])
        self.opt_energy = None
        self.opt_energy_regularizer = None
        self.opt_energy_intensity = None
        self.energy_threshold = 1e-3
        self.gradient_norm_threshold = 1e-3

        energies = []

        # define vector fields
        initial_velocity_field = np.zeros((self.dim, *self.shape), dtype=np.double)
        velocity_fields = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)
        gradient_initial_velocity = np.copy(initial_velocity_field)

        with self.logger.block("Perform image matching via geodesic shooting ..."):
            try:
                for k in range(iterations):
                    # update estimate of initial velocity
                    initial_velocity_field -= epsilon * gradient_initial_velocity

                    # integrate initial velocity field forward in time
                    velocity_fields = self.integrate_forward_vector_field(initial_velocity_field)

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
                    energy_intensity = 1 / sigma**2 * compute_energy(forward_pushed_input)
                    energy = energy_regularizer + energy_intensity

                    # compute the norm of the gradient; stop if below threshold
                    # (updates are too small)
                    norm_gradient_initial_velocity = np.linalg.norm(gradient_initial_velocity)
                    if (norm_gradient_initial_velocity < self.gradient_norm_threshold
                       and energy > self.energy_threshold):
                        self.logger.warning(f"Gradient norm is {norm_gradient_initial_velocity} "
                                            "and therefore below threshold. Stopping ...")
                        break

                    # stop if energy is below threshold
                    if energy < self.energy_threshold:
                        self.opt_energy = energy
                        self.opt_energy_regularizer = energy_regularizer
                        self.opt_energy_intensity = energy_intensity
                        self.opt = (forward_pushed_input, initial_velocity_field, energies, flow)
                        self.logger.info(f"Energy below threshold of {self.energy_threshold}. "
                                         "Stopping ...")
                        break

                    # update optimal energy if necessary
                    if self.opt_energy is None or energy < self.opt_energy:
                        self.opt_energy = energy
                        self.opt_energy_regularizer = energy_regularizer
                        self.opt_energy_intensity = energy_intensity
                        self.opt = (forward_pushed_input, initial_velocity_field, energies, flow)

                    energies.append(energy)

                    # output of current iteration and energies
                    self.logger.info(f"iter: {k:3d}, "
                                     f"energy: {energy:4.2f}, "
                                     f"L2: {energy_intensity:4.2f}, "
                                     f"reg: {energy_regularizer:4.2f}")
            except KeyboardInterrupt:
                self.logger.warning("Aborting registration ...")

        self.logger.info("Finished registration ...")

        if self.opt_energy is not None:
            self.logger.info(f"Optimal energy: {self.opt_energy:4.4f}")
            self.logger.info(f"Optimal intensity difference: {self.opt_energy_intensity:4.4f}")
            self.logger.info(f"Optimal regularization: {self.opt_energy_regularizer:4.4f}")

        if self.opt[1] is not None:
            # compute the length of the path on the manifold;
            # this step only requires the initial velocity due to conservation of momentum
            length = np.linalg.norm(self.regularizer.cauchy_navier(self.opt[1]))
        else:
            length = 0.0

        if return_all:
            return self.opt + (length,)
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

        for time in range(0, self.time_steps-1):
            alpha = self.forward_alpha(velocity_fields[time])
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
        velocity_fields[0] = initial_velocity_field

        einsum_string = 'kl...,l...->k...'
        einsum_string_transpose = 'lk...,l...->k...'

        for time in range(0, self.time_steps-2):
            momentum_t = self.regularizer.cauchy_navier(velocity_fields[time])
            grad_mt = finite_difference(momentum_t)
            grad_vt = finite_difference(velocity_fields[time])
            div_vt = np.sum(np.array([grad_vt[d, d, ...] for d in range(self.dim)]), axis=0)
            rhs = (np.einsum(einsum_string_transpose, grad_vt, momentum_t)
                   + np.einsum(einsum_string, grad_mt, velocity_fields[time])
                   + momentum_t * div_vt[np.newaxis, ...])
            velocity_fields[time+1] = (velocity_fields[time]
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

        for time in range(self.time_steps-2, -1, -1):
            grad_velocity_fields = finite_difference(velocity_fields[time])
            div_velocity_fields = np.sum(np.array([grad_velocity_fields[d, d, ...]
                                                   for d in range(self.dim)]), axis=0)
            regularized_v = self.regularizer.cauchy_navier(v_old)
            grad_regularized_v = finite_difference(regularized_v)
            rhs_v = - self.regularizer.cauchy_navier_squared_inverse(
                np.einsum(einsum_string_transpose, grad_velocity_fields, regularized_v)
                + np.einsum(einsum_string, grad_regularized_v, velocity_fields[time])
                + regularized_v * div_velocity_fields[np.newaxis, ...])
            v_old = v_old - rhs_v / self.time_steps

            grad_delta_v = finite_difference(delta_v)
            div_delta_v = np.sum(np.array([grad_delta_v[d, d, ...]
                                           for d in range(self.dim)]), axis=0)
            regularized_velocity_fields = self.regularizer.cauchy_navier(velocity_fields[time])
            grad_regularized_velocity_fields = finite_difference(regularized_velocity_fields)
            rhs_delta_v = (- v_old
                           - (np.einsum(einsum_string, grad_velocity_fields, delta_v)
                              - np.einsum(einsum_string, grad_delta_v, velocity_fields[time]))
                           + self.regularizer.cauchy_navier_squared_inverse(
                               np.einsum(einsum_string_transpose, grad_delta_v,
                                         regularized_velocity_fields)
                               + np.einsum(einsum_string, grad_regularized_velocity_fields, delta_v)
                               + regularized_velocity_fields * div_delta_v[np.newaxis, ...]))
            delta_v = delta_v_old - rhs_delta_v / self.time_steps
            delta_v_old = delta_v

        return delta_v

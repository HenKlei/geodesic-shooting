import time
import numpy as np

from geodesic_shooting.utils import sampler, grid
from geodesic_shooting.utils.grad import finite_difference
from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


class LDDMM:
    """Class that implements the original large deformation metric mappings algorithm.

    Based on:
    Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms.
    Beg, Miller, Trouvé, Younes, 2004
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
        self.energy_threshold = 1e-3
        self.gradient_norm_threshold = 1e-3

        self.logger = getLogger('lddmm', level=log_level)

    def register(self, input_, target, time_steps=30, iterations=1000, sigma=1, epsilon=0.01,
                 early_stopping=10, return_all=False):
        """Performs actual registration according to LDDMM algorithm with time-varying velocity
           fields that can be chosen independently of each other (respecting smoothness assumption).

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
        early_stopping
            Number of iterations with non-decreasing energy after which to stop registration.
            If `None`, no early stopping is used.
        return_all
            Determines whether or not to return all information or only the final flow that led to
            the best registration result.

        Returns
        -------
        Either the best flow (if return_all is True) or a tuple consisting of the registered image,
        the velocities, the energies, the flows and inverse flows, the forward-pushed input and the
        back-pulled target at all time instances.
        """
        assert isinstance(time_steps, int) and time_steps > 0
        assert isinstance(iterations, int) and iterations > 0
        assert sigma > 0
        assert 0 < epsilon < 1
        assert (isinstance(early_stopping, int) and early_stopping > 0) or early_stopping is None
        assert input_.shape == target.shape

        input_ = input_.astype('double')
        target = target.astype('double')

        def compute_energy(image):
            return np.sum((image - target)**2)

        def compute_grad_energy(determinants, image_gradient, image, image_target):
            return self.regularizer.cauchy_navier_squared_inverse(
                2 * determinants[np.newaxis, ...] * image_gradient
                * (image - image_target)[np.newaxis, ...])

        # set up variables
        self.time_steps = time_steps
        self.shape = input_.shape
        self.dim = input_.ndim

        self.energy_threshold = 1e-3

        energy_not_decreasing = 0

        # define vector fields
        velocity_fields = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)
        gradient_velocity_fields = np.copy(velocity_fields)

        def set_opt(opt, energy, energy_regularizer, energy_intensity, energy_intensity_unscaled,
                    transformed_input, forward_pushed_input, back_pulled_target,
                    velocity_fields, forward_flows, backward_flows):
            opt['energy'] = energy
            opt['energy_regularizer'] = energy_regularizer
            opt['energy_intensity'] = energy_intensity
            opt['energy_intensity_unscaled'] = energy_intensity_unscaled
            opt['transformed_input'] = transformed_input
            opt['forward_pushed_input'] = forward_pushed_input
            opt['back_pulled_target'] = back_pulled_target
            opt['velocity_fields'] = velocity_fields
            opt['forward_flows'] = forward_flows
            opt['backward_flows'] = backward_flows
            return opt

        opt = set_opt({}, None, None, None, None, input_, input_, target, velocity_fields,
                      self.integrate_backward_flow(velocity_fields),
                      self.integrate_forward_flow(velocity_fields))

        k = 0
        reason_registration_ended = 'reached maximum number of iterations'

        start_time = time.perf_counter()

        with self.logger.block("Perform image matching via LDDMM algorithm ..."):
            try:
                while k < iterations:
                    # update estimate of velocity
                    velocity_fields -= epsilon * gradient_velocity_fields

                    # reparameterize velocity fields
                    if k % 10 == 9:
                        velocity_fields = self.reparameterize(velocity_fields)

                    # compute backward flows
                    backward_flows = self.integrate_backward_flow(velocity_fields)

                    # compute forward flows
                    forward_flows = self.integrate_forward_flow(velocity_fields)

                    # push-forward input_ image
                    forward_pushed_input = self.push_forward(input_, forward_flows)

                    # pull-back target image
                    back_pulled_target = self.pull_back(target, backward_flows)

                    # compute gradient of forward-pushed input_ image
                    gradient_forward_pushed_input = self.image_grad(forward_pushed_input)

                    # compute Jacobian determinant of the transformation
                    det_backward_flows = self.jacobian_determinant(backward_flows)
                    if self.is_injectivity_violated(det_backward_flows):
                        self.logger.error("Injectivity violated. Stopping. "
                                          "Try lowering the learning rate (epsilon).")
                        break

                    # compute the gradient of the energy
                    for t in range(0, self.time_steps):
                        grad = compute_grad_energy(det_backward_flows[t],
                                                   gradient_forward_pushed_input[t],
                                                   forward_pushed_input[t],
                                                   back_pulled_target[t])
                        gradient_velocity_fields[t] = (2 * velocity_fields[t] - 1 / sigma**2 * grad)

                    # compute the norm of the gradient of the energy; stop if small
                    norm_gradient_velocity_fields = np.linalg.norm(gradient_velocity_fields)
                    if norm_gradient_velocity_fields < self.gradient_norm_threshold:
                        self.logger.warning(f"Gradient norm is {norm_gradient_velocity_fields} "
                                            "and therefore below threshold. Stopping ...")
                        reason_registration_ended = 'reached gradient norm threshold'
                        break

                    # compute the current energy consisting of intensity difference
                    # and regularization
                    energy_regularizer = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(
                        velocity_fields[t])) for t in range(self.time_steps)])
                    energy_intensity_unscaled = compute_energy(forward_pushed_input[-1])
                    energy_intensity = 1 / sigma**2 * energy_intensity_unscaled
                    energy = energy_regularizer + energy_intensity

                    # stop if energy is below threshold
                    if energy < self.energy_threshold:
                        opt = set_opt(opt, energy, energy_regularizer, energy_intensity,
                                      energy_intensity_unscaled, forward_pushed_input[-1],
                                      forward_pushed_input, back_pulled_target,
                                      velocity_fields, forward_flows, backward_flows)
                        self.logger.warning(f"Energy below threshold of {self.energy_threshold}. "
                                            "Stopping ...")
                        reason_registration_ended = 'reached energy threshold'
                        break

                    # update optimal energy if necessary
                    if opt['energy'] is None or energy < opt['energy']:
                        opt = set_opt(opt, energy, energy_regularizer, energy_intensity,
                                      energy_intensity_unscaled, forward_pushed_input[-1],
                                      forward_pushed_input, back_pulled_target,
                                      velocity_fields, forward_flows, backward_flows)
                        energy_not_decreasing = 0
                    else:
                        energy_not_decreasing += 1

                    if early_stopping is not None and energy_not_decreasing >= early_stopping:
                        self.logger.info(f"Energy did not decrease for {early_stopping} "
                                         "iterations. Early stopping ...")
                        reason_registration_ended = 'early stopping due to non-decreasing energy'
                        break

                    k += 1

                    # output of current iteration and energies
                    self.logger.info(f"iter: {k:3d}, "
                                     f"energy: {energy:4.2f}, "
                                     f"L2 (w/o scale): {energy_intensity_unscaled:4.4f}, "
                                     f"reg: {energy_regularizer:4.2f}")
            except KeyboardInterrupt:
                self.logger.warning("Aborting registration ...")
                reason_registration_ended = 'manual abort'

        elapsed_time = int(time.perf_counter() - start_time)

        self.logger.info("Finished registration ...")

        if opt['energy'] is not None:
            self.logger.info(f"Optimal energy: {opt['energy']:4.4f}")
            self.logger.info("Optimal intensity difference (with scale): "
                             f"{opt['energy_intensity']:4.4f}")
            self.logger.info("Optimal intensity difference (without scale): "
                             f"{opt['energy_intensity_unscaled']:4.4f}")
            self.logger.info(f"Optimal regularization: {opt['energy_regularizer']:4.4f}")

        if opt['velocity_fields'] is not None:
            # compute the length of the path on the manifold
            length = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(
                                 opt['velocity_fields'][t]))
                             for t in range(self.time_steps)])
        else:
            length = 0.0

        opt['length'] = length
        opt['iterations'] = k
        opt['time'] = elapsed_time
        opt['reason_registration_ended'] = reason_registration_ended

        if return_all:
            return opt
        return forward_flows[-1]

    def reparameterize(self, velocity_fields):
        """Reparameterizes velocity fields to obtain a time-dependent velocity field
           with constant speed.

        Parameters
        ----------
        velocity_field
            Sequence of velocity fields (i.e. time-depending velocity field).

        Returns
        -------
        Array consisting of reparametrized velocity fields.
        """
        length = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(velocity_fields[t]))
                         for t in range(self.time_steps)])
        for t in range(self.time_steps):
            velocity_fields[t] = (length / self.time_steps * velocity_fields[t]
                                  / np.linalg.norm(self.regularizer.cauchy_navier(
                                        velocity_fields[t])))
        return velocity_fields

    def integrate_backward_flow(self, velocity_fields):
        """Computes backward integration according to given velocity fields.

        Parameters
        ----------
        velocity_fields
            Sequence of velocity fields (i.e. time-depending velocity field).

        Returns
        -------
        Array containing the flows at different time instances.
        """
        # make identity grid
        identity_grid = grid.coordinate_grid(self.shape)

        # create flow
        flows = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)

        # final flow is the identity mapping
        flows[self.time_steps-1] = identity_grid

        # perform backward integration
        for t in range(self.time_steps-2, -1, -1):
            alpha = self.backwards_alpha(velocity_fields[t])
            flows[t] = sampler.sample(flows[t+1], identity_grid + alpha)

        return flows

    def backwards_alpha(self, velocity_field):
        """Helper function to estimate the updated positions (backward calculation).

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
            alpha = sampler.sample(velocity_field, identity_grid + 0.5 * alpha)
        return alpha

    def integrate_forward_flow(self, velocity_fields):
        """Computes forward integration according to given velocity fields.

        Parameters
        ----------
        velocity_fields
            Sequence of velocity fields (i.e. time-depending velocity field).

        Returns
        -------
        Array containing the flows at different time instances.
        """
        # make identity grid
        identity_grid = grid.coordinate_grid(self.shape)

        # create flow
        flows = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)

        # initial flow is the identity mapping
        flows[0] = identity_grid

        # perform forward integration
        for t in range(0, self.time_steps-1):
            alpha = self.forward_alpha(velocity_fields[t])
            flows[t+1] = sampler.sample(flows[t], identity_grid - alpha)

        return flows

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
            Array containing a sequence of flows along which to push the input forward.

        Returns
        -------
        Array consisting of a sequence of forward-pushed images.
        """
        result = np.zeros((self.time_steps,) + image.shape, dtype=np.double)
        for t in range(0, self.time_steps):
            result[t] = sampler.sample(image, flow[t])
        return result

    def pull_back(self, image, flows):
        """Pulls back an image along a flow.

        Parameters
        ----------
        image
            Array to pull back.
        flows
            Array containing a sequence of flows along which to pull the input back.

        Returns
        -------
        Array consisting of a sequence of back-pulled images.
        """
        result = np.zeros((self.time_steps,) + image.shape, dtype=np.double)
        for t in range(self.time_steps-1, -1, -1):
            result[t] = sampler.sample(image, flows[t])
        return result

    def image_grad(self, images):
        """Computes the gradients of the given images.

        Parameters
        ----------
        images
            Array containing a sequence of (forward) pushed images.

        Returns
        -------
        Array consisting of a sequence of gradients of the input images.
        """
        gradients = np.zeros((images.shape[0], self.dim, *images.shape[1:]), dtype=np.double)
        for t in range(self.time_steps):
            gradients[t] = finite_difference(images[t])
        return gradients

    def jacobian_determinant(self, transformations):
        """Computes the determinant of the Jacobian for a sequence of transformations at each point.

        implements step (8): Calculate Jacobian determinant of the transformation.
        Parameters
        ----------
        transformations
            Array consisting of a sequence of transformations.

        Returns
        -------
        Array of determinants.
        """
        determinants = np.zeros((self.time_steps, *self.shape), dtype=np.double)

        for t in range(self.time_steps):
            if self.dim == 1:
                grad_x = finite_difference(transformations[t, 0, ...])

                determinants[t] = grad_x[0, ...]
            elif self.dim == 2:
                # get gradient in x-direction
                grad_x = finite_difference(transformations[t, 0, ...])
                # gradient in y-direction
                grad_y = finite_difference(transformations[t, 1, ...])

                # calculate determinants
                determinants[t] = (grad_x[0, ...] * grad_y[1, ...]
                                   - grad_x[1, ...] * grad_y[0, ...])
            elif self.dim == 3:
                # get gradient in x-direction
                grad_x = finite_difference(transformations[t, 0, ...])
                # gradient in y-direction
                grad_y = finite_difference(transformations[t, 1, ...])
                # gradient in z-direction
                grad_z = finite_difference(transformations[t, 2, ...])

                # calculate determinants
                determinants[t] = (grad_x[0, ...] * grad_y[1, ...] * grad_z[2, ...]
                                   + grad_y[0, ...] * grad_z[1, ...] * grad_x[2, ...]
                                   + grad_z[0, ...] * grad_x[1, ...] * grad_y[2, ...]
                                   - grad_x[2, ...] * grad_y[1, ...] * grad_z[0, ...]
                                   - grad_y[2, ...] * grad_z[1, ...] * grad_x[0, ...]
                                   - grad_z[2, ...] * grad_x[1, ...] * grad_y[0, ...])

        return determinants

    def is_injectivity_violated(self, determinants):
        """Checks injectivity and orientation preservation by considering determinants.

        A function has a differentiable inverse and preserves orientation if and only if
        the determinant of its jacobian is positive.

        Parameters
        ----------
        determinants
            Sequence of determinants to check.

        Returns
        -------
        True if any value in the input is negative, False otherwise.
        """
        return (determinants < 0).any()

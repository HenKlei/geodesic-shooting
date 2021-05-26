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
        self.opt = None
        self.opt_energy = None
        self.opt_energy_regularizer = None
        self.opt_energy_intensity = None
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
        self.opt = ()
        self.opt_energy = None
        self.opt_energy_regularizer = None
        self.opt_energy_intensity = None
        self.energy_threshold = 1e-3

        energies = []

        energy_not_decreasing = 0

        # define vector fields
        velocity_fields = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)
        gradient_velocity_fields = np.copy(velocity_fields)

        with self.logger.block("Perform image matching via LDDMM algorithm ..."):
            try:
                for k in range(iterations):
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
                    for time in range(0, self.time_steps):
                        grad = compute_grad_energy(det_backward_flows[time],
                                                   gradient_forward_pushed_input[time],
                                                   forward_pushed_input[time],
                                                   back_pulled_target[time])
                        gradient_velocity_fields[time] = (2 * velocity_fields[time]
                                                          - 1 / sigma**2 * grad)

                    # compute the norm of the gradient of the energy; stop if small
                    norm_gradient_velocity_fields = np.linalg.norm(gradient_velocity_fields)
                    if norm_gradient_velocity_fields < self.gradient_norm_threshold:
                        self.logger.warning(f"Gradient norm is {norm_gradient_velocity_fields} "
                                            "and therefore below threshold. Stopping ...")
                        break

                    # compute the current energy consisting of intensity difference
                    # and regularization
                    energy_regularizer = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(
                        velocity_fields[time])) for time in range(self.time_steps)])
                    energy_intensity = 1 / sigma**2 * compute_energy(forward_pushed_input[-1])
                    energy = energy_regularizer + energy_intensity

                    # stop if energy is below threshold
                    if energy < self.energy_threshold:
                        self.opt_energy = energy
                        self.opt_energy_regularizer = energy_regularizer
                        self.opt_energy_intensity = energy_intensity
                        self.opt = (forward_pushed_input[-1], velocity_fields, energies,
                                    forward_flows, backward_flows, forward_pushed_input,
                                    back_pulled_target)
                        self.logger.warning(f"Energy below threshold of {self.energy_threshold}. "
                                            "Stopping ...")
                        break

                    # update optimal energy if necessary
                    if self.opt_energy is None or energy < self.opt_energy:
                        self.opt_energy = energy
                        self.opt_energy_regularizer = energy_regularizer
                        self.opt_energy_intensity = energy_intensity
                        self.opt = (forward_pushed_input[-1], velocity_fields, energies,
                                    forward_flows, backward_flows, forward_pushed_input,
                                    back_pulled_target)
                        energy_not_decreasing = 0
                    else:
                        energy_not_decreasing += 1

                    if early_stopping is not None and energy_not_decreasing >= early_stopping:
                        self.logger.info(f"Energy did not decrease for {early_stopping} "
                                         "iterations. Early stopping ...")
                        break

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
            # compute the length of the path on the manifold
            length = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(self.opt[1][time]))
                             for time in range(self.time_steps)])
        else:
            length = 0.0

        if return_all:
            return self.opt + (length,)
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
        length = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(velocity_fields[time]))
                         for time in range(self.time_steps)])
        for time in range(self.time_steps):
            velocity_fields[time] = (length / self.time_steps * velocity_fields[time]
                                     / np.linalg.norm(self.regularizer.cauchy_navier(
                                         velocity_fields[time])))
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
        for time in range(self.time_steps-2, -1, -1):
            alpha = self.backwards_alpha(velocity_fields[time])
            flows[time] = sampler.sample(flows[time+1], identity_grid + alpha)

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
        for time in range(0, self.time_steps-1):
            alpha = self.forward_alpha(velocity_fields[time])
            flows[time+1] = sampler.sample(flows[time], identity_grid - alpha)

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
        for time in range(0, self.time_steps):
            result[time] = sampler.sample(image, flow[time])
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
        for time in range(self.time_steps-1, -1, -1):
            result[time] = sampler.sample(image, flows[time])
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
        for time in range(self.time_steps):
            gradients[time] = finite_difference(images[time])
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

        for time in range(self.time_steps):
            if self.dim == 1:
                grad_x = finite_difference(transformations[time, 0, ...])

                determinants[time] = grad_x[0, ...]
            elif self.dim == 2:
                # get gradient in x-direction
                grad_x = finite_difference(transformations[time, 0, ...])
                # gradient in y-direction
                grad_y = finite_difference(transformations[time, 1, ...])

                # calculate determinants
                determinants[time] = (grad_x[0, ...] * grad_y[1, ...]
                                      - grad_x[1, ...] * grad_y[0, ...])
            elif self.dim == 3:
                # get gradient in x-direction
                grad_x = finite_difference(transformations[time, 0, ...])
                # gradient in y-direction
                grad_y = finite_difference(transformations[time, 1, ...])
                # gradient in z-direction
                grad_z = finite_difference(transformations[time, 2, ...])

                # calculate determinants
                determinants[time] = (grad_x[0, ...] * grad_y[1, ...] * grad_z[2, ...]
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

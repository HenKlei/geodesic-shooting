import numpy as np

from geodesic_shooting.utils import sampler, grid
from geodesic_shooting.utils.grad import finite_difference
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


class LDDMM:
    """Class that implements the original large deformation metric mappings algorithm.

    Based on:
    Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms.
    Beg, Miller, Trouvé, Younes, 2004
    """
    def __init__(self, alpha=6., gamma=1.):
        """Constructor.

        Parameters
        ----------
        alpha
            Parameter for biharmonic regularizer.
        gamma
            Parameter for biharmonic regularizer.
        """
        self.regularizer = BiharmonicRegularizer(alpha, gamma)

    def register(self, input_, target, time_steps=30, iterations=100, sigma=1, epsilon=0.01, return_all=False):
        """Performs actual registration according to LDDMM algorithm with time-varying velocity fields that can be chosen independently of each other (respecting smoothness assumption).

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
            Weight for the similarity measurement (L2 difference of the target and the registered image); the smaller sigma, the larger the influence of the L2 loss.
        epsilon
            Learning rate, i.e. step size of the optimizer.
        return_all
            Determines whether or not to return all information or only the final flow that led to the best registration result.

        Returns
        -------
        Either the best flow (if return_all is True) or a tuple consisting of the registered image,the velocities, the energies, the flows and inverse flows, the forward-pushed input and the back-pulled target at all time instances.
        """
        assert isinstance(time_steps, int) and time_steps > 0
        assert isinstance(iterations, int) and iterations > 0
        assert sigma > 0
        assert 0 < epsilon < 1
        assert input_.shape == target.shape

        input_ = input_.astype('double')
        target = target.astype('double')

        def energy(J0):
            return np.sum((J0 - target)**2)

        def grad_energy(detPhi1, dJ0, J0, J1):
            return self.regularizer.cauchy_navier_squared_inverse(2 * detPhi1[np.newaxis, ...] * dJ0 * (J0 - J1)[np.newaxis, ...])

        # set up variables
        self.time_steps = time_steps
        self.shape = input_.shape
        self.dim = input_.ndim
        self.opt = ()
        self.E_opt = None
        self.energy_threshold = 1e-3

        energies = []

        # define vector fields
        v = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)
        dv = np.copy(v)

        # (12): iteration over k
        for k in range(iterations):

            # (1): Calculate new estimate of velocity
            v -= epsilon * dv

            # (2): Reparameterize
            if k % 10 == 9:
                v = self.reparameterize(v)

            # (3): calculate backward flows
            Phi1 = self.integrate_backward_flow(v)

            # (4): calculate forward flows
            Phi0 = self.integrate_forward_flow(v)

            # (5): push-forward input_
            J0 = self.push_forward(input_, Phi0)

            # (6): pull back target
            J1 = self.pull_back(target, Phi1)

            # (7): Calculate image gradient
            dJ0 = self.image_grad(J0)

            # (8): Calculate Jacobian determinant of the transformation
            detPhi1 = self.jacobian_determinant(Phi1)
            if self.is_injectivity_violated(detPhi1):
                print("Injectivity violated. Stopping. Try lowering the learning rate (epsilon).")
                break

            # (9): Calculate the gradient
            for t in range(0, self.time_steps):
                grad = grad_energy(detPhi1[t], dJ0[t], J0[t], J1[t])
                dv[t] = 2*v[t] - 1 / sigma**2 * grad

            # (10) calculate norm of the gradient, stop if small
            dv_norm = np.linalg.norm(dv)
            if dv_norm < 0.001:
                print(f"Gradient norm is {dv_norm} and therefore below threshold. Stopping ...")
                break

            # (11): calculate new energy
            E_regularizer = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(v[t])) for t in range(self.time_steps)])
            E_intensity = 1 / sigma**2 * energy(J0[-1])
            E = E_regularizer + E_intensity

            if E < self.energy_threshold:
                self.E_opt = E
                self.opt = (J0[-1], v, energies, Phi0, Phi1, J0, J1)
                print(f"Energy below threshold of {self.energy_threshold}. Stopping ...")
                break

            if self.E_opt is None or E < self.E_opt:
                self.E_opt = E
                self.opt = (J0[-1], v, energies, Phi0, Phi1, J0, J1)

            energies.append(E)

            # (12): iterate k = k+1
            print("iteration {:3d}, energy {:4.2f}, thereof {:4.2f} regularization and {:4.2f} intensity difference".format(k, E, E_regularizer, E_intensity))
            # end of for loop block

        print("Optimal energy {:4.2f}".format(self.E_opt))

        # (13): Denote the final velocity field as \hat{v}
        v_hat = self.opt[1]

        # (14): Calculate the length of the path on the manifold
        length = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(v_hat[t])) for t in range(self.time_steps)])

        if return_all:
            return self.opt + (length,)
        else:
            return Phi0[-1]

    def reparameterize(self, velocity_fields):
        """Reparameterizes velocity fields to obtain a time-dependent velocity field with constant speed.

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
            velocity_fields[t] = length / self.time_steps * velocity_fields[t] / np.linalg.norm(self.regularizer.cauchy_navier(velocity_fields[t]))
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
        flows[self.time_steps - 1] = identity_grid

        # perform backward integration
        for t in range(self.time_steps-2, -1, -1):
            alpha = self.backwards_alpha(velocity_fields[t])
            flows[t] = sampler.sample(flows[t + 1], identity_grid + alpha)

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
        """Pushs forward an image along a flow.

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
                dx = finite_difference(transformations[t, 0, ...])

                determinants[t] = dx[0, ...]
            elif self.dim == 2:
                # get gradient in x-direction
                dx = finite_difference(transformations[t, 0, ...])
                # gradient in y-direction
                dy = finite_difference(transformations[t, 1, ...])

                # calculate determinants
                determinants[t] = dx[0, ...] * dy[1, ...] - dx[1, ...] * dy[0, ...]
            elif self.dim == 3:
                # get gradient in x-direction
                dx = finite_difference(transformations[t, 0, ...])
                # gradient in y-direction
                dy = finite_difference(transformations[t, 1, ...])
                # gradient in z-direction
                dz = finite_difference(transformations[t, 2, ...])

                # calculate determinants
                determinants[t] = (dx[0, ...] * dy[1, ...] * dz[2, ...]
                                   + dy[0, ...] * dz[1, ...] * dx[2, ...]
                                   + dz[0, ...] * dx[1, ...] * dy[2, ...]
                                   - dx[2, ...] * dy[1, ...] * dz[0, ...]
                                   - dy[2, ...] * dz[1, ...] * dx[0, ...]
                                   - dz[2, ...] * dx[1, ...] * dy[0, ...])

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

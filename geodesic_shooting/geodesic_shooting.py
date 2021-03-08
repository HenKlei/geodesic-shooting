import numpy as np

from geodesic_shooting.utils import sampler, grid
from geodesic_shooting.utils.grad import finite_difference
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


class GeodesicShooting:
    """Class that implements large deformation metric mappings via geodesic shooting.

    Based on:
    Geodesic Shooting for Computational Anatomy.
    Miller, Trouvé, Younes, 2006
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
        """Performs actual registration according to LDDMM algorithm with time-varying velocity fields that are chosen via geodesics.

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
        Either the best flow (if return_all is True) or a tuple consisting of the registered image, the velocities, the energies, the flows and inverse flows, the forward-pushed input and the back-pulled target at all time instances.
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

        def grad_energy(dJ0, J0):
            return self.regularizer.cauchy_navier_squared_inverse(dJ0 * (J0 - target)[np.newaxis, ...])

        # set up variables
        self.time_steps = time_steps
        self.shape = input_.shape
        self.dim = input_.ndim
        self.opt = (input_, None,)
        self.opt += ([],) * 2
        self.E_opt = None
        self.energy_threshold = 1e-3

        energies = []

        # define vector fields
        initial_velocity_field = np.zeros((self.dim, *self.shape), dtype=np.double)
        velocity_fields = np.zeros((self.time_steps, self.dim, *self.shape), dtype=np.double)
        dinitial_velocity_field = np.copy(initial_velocity_field)

        # (12): iteration over k
        for k in range(iterations):

            # (1): Calculate new estimate of velocity
            initial_velocity_field -= epsilon * dinitial_velocity_field

            velocity_fields = self.integrate_forward_vector_field(initial_velocity_field)

            # (4): calculate forward flows
            Phi0 = self.integrate_forward_flow(velocity_fields)

            # (5): push-forward input_
            J0 = self.push_forward(input_, Phi0)

            # (7): Calculate image gradient
            dJ0 = self.image_grad(J0)

            gradient_L2_energy = 1 / sigma**2 * grad_energy(dJ0, J0)

            dinitial_velocity_field = - self.integrate_backward_adjoint_Jacobi_field_equations(gradient_L2_energy, velocity_fields)

            # (10) calculate norm of the gradient, stop if small
            dinitial_velocity_field_norm = np.linalg.norm(dinitial_velocity_field)
            if dinitial_velocity_field_norm < 0.001:
                print(f"Gradient norm is {dinitial_velocity_field_norm} and therefore below threshold. Stopping ...")
                break

            # (11): calculate new energy
            E_regularizer = np.linalg.norm(self.regularizer.cauchy_navier(initial_velocity_field))
            E_intensity = 1 / sigma**2 * energy(J0)
            E = E_regularizer + E_intensity

            if E < self.energy_threshold:
                self.E_opt = E
                self.opt = (J0, initial_velocity_field, energies, Phi0)
                print(f"Energy below threshold of {self.energy_threshold}. Stopping ...")
                break

            if self.E_opt is None or E < self.E_opt:
                self.E_opt = E
                self.opt = (J0, initial_velocity_field, energies, Phi0)

            energies.append(E)

            # (12): iterate k = k+1
            print("iteration {:3d}, energy {:4.2f}, thereof {:4.2f} regularization and {:4.2f} intensity difference".format(k, E, E_regularizer, E_intensity))
            # end of for loop block

        if self.E_opt is not None:
            print("Optimal energy {:4.2f}".format(self.E_opt))

        # (13): Denote the final velocity field as \hat{v}
        v_hat = self.opt[1]

        if v_hat is not None:
            # (14): Calculate the length of the path on the manifold
            length = np.linalg.norm(self.regularizer.cauchy_navier(v_hat))
        else:
            length = 0.0

        if return_all:
            return self.opt + (length,)
        else:
            return Phi0

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

        # Phi0_0 is the identity mapping
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
        """Pushs forward an image along a flow.

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

        for t in range(0, self.time_steps-2):
            mt = self.regularizer.cauchy_navier(velocity_fields[t])
            grad_mt = finite_difference(mt)[0:self.dim, ...]
            grad_vt = finite_difference(velocity_fields[t])[0:self.dim, ...]
            div_vt = np.sum(np.array([grad_vt[d, d, ...] for d in range(self.dim)]), axis=0)
            rhs = np.einsum(einsum_string_transpose, grad_vt, mt) + np.einsum(einsum_string, grad_mt, velocity_fields[t]) + mt * div_vt[np.newaxis, ...]
            velocity_fields[t+1] = velocity_fields[t] - self.regularizer.cauchy_navier_squared_inverse(rhs) / self.time_steps

        return velocity_fields

    def integrate_backward_adjoint_Jacobi_field_equations(self, gradient_L2_energy, velocity_fields):
        """Performs backward integration of the adjoint jacobi field equations.

        Parameters
        ----------
        gradient_L2_energy
            Array containing the gradient of the L2 energy functional.
        velocity_fields
            Sequence of velocity fields (i.e. time-dependent velocity field) to integrate backwards.

        Returns
        -------
        Gradient of the energy with respect to the initial velocity field.
        """
        v_old = gradient_L2_energy
        delta_v_old = np.zeros(v_old.shape, dtype=np.double)
        delta_v = delta_v_old

        einsum_string = 'kl...,l...->k...'
        einsum_string_transpose = 'lk...,l...->k...'

        for t in range(self.time_steps-2, -1, -1):
            grad_velocity_fields = finite_difference(velocity_fields[t])[0:self.dim, ...]
            div_velocity_fields = np.sum(np.array([grad_velocity_fields[d, d, ...] for d in range(self.dim)]), axis=0)
            Lv = self.regularizer.cauchy_navier(v_old)
            grad_Lv = finite_difference(Lv)[0:self.dim, ...]
            rhs_v = - self.regularizer.cauchy_navier_squared_inverse(np.einsum(einsum_string_transpose, grad_velocity_fields, Lv) + np.einsum(einsum_string, grad_Lv, velocity_fields[t]) + Lv * div_velocity_fields[np.newaxis, ...])
            v_old = v_old - rhs_v / self.time_steps

            grad_delta_v = finite_difference(delta_v)[0:self.dim, ...]
            div_delta_v = np.sum(np.array([grad_delta_v[d, d, ...] for d in range(self.dim)]), axis=0)
            Lvelocity_fields = self.regularizer.cauchy_navier(velocity_fields[t])
            grad_Lvelocity_fields = finite_difference(Lvelocity_fields)[0:self.dim, ...]
            rhs_delta_v = - v_old - (np.einsum(einsum_string, grad_velocity_fields, delta_v) - np.einsum(einsum_string, grad_delta_v, velocity_fields[t])) + self.regularizer.cauchy_navier_squared_inverse(np.einsum(einsum_string_transpose, grad_delta_v, Lvelocity_fields) + np.einsum(einsum_string, grad_Lvelocity_fields, delta_v) + Lvelocity_fields * div_delta_v[np.newaxis, ...])
            delta_v = delta_v_old - rhs_delta_v / self.time_steps
            delta_v_old = delta_v

        return delta_v

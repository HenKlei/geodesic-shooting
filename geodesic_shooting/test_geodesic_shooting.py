import numpy as np

from geodesic_shooting.utils import grid, sampler
from geodesic_shooting.utils.grad import (divergence_matrix, finite_difference_matrix,
                                          gradient_matrix)
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer
from geodesic_shooting.utils.helper_functions import tuple_product


class TestGeodesicShooting:
    """Class that implements large deformation metric mappings via geodesic shooting.

    Based on:
    Geodesic Shooting for Computational Anatomy.
    Miller, Trouvé, Younes, 2006
    """
    def __init__(self, alpha=6., exponent=1):
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
        self.size = None
        self.dim = None
        self.opt = None
        self.opt_energy = None
        self.energy_threshold = 1e-3
        self.gradient_norm_threshold = 1e-3

        self.matrices_forward_integration = []
        self.matrices_backward_integration_1 = []
        self.matrices_backward_integration_2 = []
        self.matrices_backward_integration_3 = []

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
            eight for the similarity measurement (L2 difference of the target and the registered
            image); the smaller sigma, the larger the influence of the L2 loss.
        epsilon
            Learning rate, i.e. step size of the optimizer.
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
        assert input_.shape == target.shape

        self.shape = input_.shape
        self.dim = input_.ndim

        input_ = input_.astype('double').flatten()
        target = target.astype('double').flatten()

        # set up variables
        self.time_steps = time_steps
        self.size = tuple_product(self.shape)
        self.opt = (input_.reshape(self.shape), None, [], [])
        self.opt_energy = None
        self.energy_threshold = 1e-3
        self.gradient_norm_threshold = 1e-3

        def compute_energy(image):
            return np.sum((image - target)**2)

        def compute_grad_energy(image_gradient, image):
            """ Not 100% sure whether this is correct... """
            return self.regularizer.cauchy_navier_inverse_matrix.dot(
                (image_gradient.reshape((self.dim, self.size))
                 * (image - target)[np.newaxis, ...]).reshape(self.dim * self.size))

        self.regularizer.init_matrices(self.shape)

        D = finite_difference_matrix(self.shape)
        assert D.shape == (self.dim * self.size, self.dim * self.size)
        div = divergence_matrix(self.shape)
        assert div.shape == (self.dim * self.size, self.dim * self.size)
        L = self.regularizer.cauchy_navier_matrix
        assert L.shape == (self.dim * self.size, self.dim * self.size)
        K = self.regularizer.cauchy_navier_inverse_matrix
        assert K.shape == (self.dim * self.size, self.dim * self.size)

        for i in range(self.dim * self.size):
            unit_vector = np.zeros(self.dim * self.size)
            unit_vector[i] = 1.
            self.matrices_forward_integration.append(-K.dot(np.diag(L[:, i]).dot(D.T)
                                                            + np.diag(unit_vector).dot(D.dot(L))
                                                            + np.diag(L[:, i]).dot(div)))
            self.matrices_backward_integration_1.append(K.dot(np.diag(L[:, i]).dot(D.T)
                                                        + np.diag(L[:, i]).dot(div)))
            self.matrices_backward_integration_2.append(K.dot(np.diag(unit_vector).dot(D.dot(L))))
            self.matrices_backward_integration_3.append(np.diag(unit_vector).dot(D))

        energies = []

        # define vector fields
        initial_velocity_field = np.zeros(self.dim * self.size, dtype=np.double)
        velocity_fields = np.zeros((self.time_steps, self.dim * self.size), dtype=np.double)
        gradient_initial_velocity = np.copy(initial_velocity_field)

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
            gradient_l2_energy = 1 / sigma**2 * compute_grad_energy(gradient_forward_pushed_input,
                                                                    forward_pushed_input)

            # compute gradient of the intensity difference with respect to the initial velocity
            gradient_initial_velocity = - self.integrate_backward_adjoint_Jacobi_field_equations(
                gradient_l2_energy, velocity_fields)

            # compute the norm of the gradient; stop if below threshold (updates are too small)
            norm_gradient_initial_velocity = np.linalg.norm(gradient_initial_velocity)
            if norm_gradient_initial_velocity < self.gradient_norm_threshold:
                print(f"Gradient norm is {norm_gradient_initial_velocity} and "
                      "therefore below threshold. Stopping ...")
                break

            # compute the current energy consisting of intensity difference and regularization
            energy_regularizer = np.linalg.norm(self.regularizer.cauchy_navier_matrix
                                                .dot(initial_velocity_field))
            energy_intensity = 1 / sigma**2 * compute_energy(forward_pushed_input)
            energy = energy_regularizer + energy_intensity

            # stop if energy is below threshold
            if energy < self.energy_threshold:
                self.opt_energy = energy
                self.opt = (forward_pushed_input.reshape(self.shape),
                            initial_velocity_field.reshape((self.dim, *self.shape)),
                            energies, flow)
                print(f"Energy below threshold of {self.energy_threshold}. Stopping ...")
                break

            # update optimal energy if necessary
            if self.opt_energy is None or energy < self.opt_energy:
                self.opt_energy = energy
                self.opt = (forward_pushed_input.reshape(self.shape),
                            initial_velocity_field.reshape((self.dim, *self.shape)),
                            energies, flow.reshape((self.dim, *self.shape)))

            energies.append(energy)

            # output of current iteration and energies
            print(f"iteration {k:3d}, energy {energy:4.2f}, "
                  f"thereof {energy_regularizer:4.2f} regularization "
                  f"and {energy_intensity:4.2f} intensity difference")

        if self.opt_energy is not None:
            print(f"Optimal energy: {self.opt_energy:4.2f}")

        if self.opt[1] is not None:
            # compute the length of the path on the manifold;
            # this step only requires the initial velocity due to conservation of momentum
            length = np.linalg.norm(self.regularizer.cauchy_navier_matrix
                                    .dot(self.opt[1].reshape(self.dim * self.size)))
        else:
            length = 0.0

        if return_all:
            return self.opt + (length,)
        return flow

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
        identity_grid = grid.coordinate_grid(self.shape).reshape(self.dim * self.size)

        # initial flow is the identity mapping
        flow = identity_grid.astype(np.double)

        for time in range(0, self.time_steps-1):
            alpha = self.forward_alpha(velocity_fields[time])
            flow = (sampler.sample(flow.reshape((self.dim, *self.shape)),
                                   (identity_grid - alpha).reshape((self.dim, *self.shape)))
                    .reshape(self.dim * self.size))

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
        identity_grid = grid.coordinate_grid(self.shape).reshape(self.dim * self.size)

        alpha = np.zeros(velocity_field.shape, dtype=np.double)
        for _ in range(5):
            alpha = (sampler.sample(velocity_field.reshape((self.dim, *self.shape)),
                                    (identity_grid - 0.5 * alpha).reshape(self.dim, *self.shape))
                     .reshape(self.dim * self.size))
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
        return sampler.sample(image.reshape(self.shape),
                              flow.reshape((self.dim, *self.shape))).reshape(self.size)

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
        assert image.shape == (self.size, )
        gradient_mat = gradient_matrix(self.shape)
        assert gradient_mat.shape == (self.dim * self.size, self.size)
        return gradient_mat.dot(image)

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
        velocity_fields = np.zeros((self.time_steps, self.dim * self.size), dtype=np.double)
        velocity_fields[0] = initial_velocity_field.reshape(self.dim * self.size)

        for time in range(0, self.time_steps-2):
            v = velocity_fields[time]
            assert v.shape == (self.dim * self.size, )
            rhs = np.sum(np.array([mat.dot(v) * v_i
                                   for mat, v_i in zip(self.matrices_forward_integration, v)]),
                         axis=0)
            assert rhs.shape == (self.dim * self.size, )
            velocity_fields[time+1] = velocity_fields[time] + rhs / self.time_steps
            assert velocity_fields[time+1].shape == (self.dim * self.size, )

        return velocity_fields

    def integrate_backward_adjoint_Jacobi_field_equations(self, gradient_l2_energy,
                                                          velocity_fields):
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
        v_adjoint = gradient_l2_energy
        assert v_adjoint.shape == (self.dim * self.size, )
        delta_v = np.zeros(v_adjoint.shape, dtype=np.double)
        assert delta_v.shape == (self.dim * self.size, )

        for time in range(self.time_steps-2, -1, -1):
            v = velocity_fields[time]
            assert v.shape == (self.dim * self.size, )
            rhs_v = - (np.sum(np.array([mat.dot(v) * delta_v_i for mat, delta_v_i in
                                        zip(self.matrices_backward_integration_1, delta_v)]),
                              axis=0) +
                       np.sum(np.array([mat.dot(delta_v) * v_i for mat, v_i in
                                        zip(self.matrices_backward_integration_2, v)]),
                              axis=0))
            assert rhs_v.shape == (self.dim * self.size, )
            v_adjoint = v_adjoint - rhs_v / self.time_steps
            assert v_adjoint.shape == (self.dim * self.size, )

            rhs_delta_v = (- v_adjoint
                           - (np.sum(np.array([mat.dot(v) * delta_v_i for mat, delta_v_i in
                                               zip(self.matrices_backward_integration_3, delta_v)]),
                                     axis=0) -
                              np.sum(np.array([mat.dot(delta_v) * v_i for mat, v_i in
                                               zip(self.matrices_backward_integration_3, v)]),
                                     axis=0))
                           + (np.sum(np.array([mat.dot(delta_v) * v_i for mat, v_i in
                                               zip(self.matrices_backward_integration_1, v)]),
                                     axis=0) +
                              np.sum(np.array([mat.dot(v) * delta_v_i for mat, delta_v_i in
                                               zip(self.matrices_backward_integration_2, delta_v)]),
                                     axis=0)))
            assert rhs_delta_v.shape == (self.dim * self.size, )
            delta_v = delta_v - rhs_delta_v / self.time_steps
            assert delta_v.shape == (self.dim * self.size, )

        return delta_v

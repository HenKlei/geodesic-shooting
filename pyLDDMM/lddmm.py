import numpy as np

from pyLDDMM.utils import sampler, grid
from pyLDDMM.utils.grad import finite_difference
from pyLDDMM.utils.regularizer import BiharmonicRegularizer


class LDDMM:
    """
    LDDMM registration; Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms.
    """
    def __init__(self, alpha=6., gamma=1.):
        self.regularizer = BiharmonicRegularizer(alpha, gamma)

    def register(self, input_, target, T=30, K=100, sigma=1, epsilon=0.01, return_all=False):
        """
        Registers two images.
        @param input_: image, ndarray.
        @param T: int, simulated discrete time steps.
        @param K: int, maximum iterations.
        @param sigma: float, sigma for L2 loss. Lower values strengthen the L2 loss.
        @param epsilon: float, learning rate.
        @return:
        """
        assert T > 0
        assert K > 0
        assert sigma > 0
        assert epsilon > 0
        assert input_.shape == target.shape

        input_ = input_.astype('double')
        target = target.astype('double')

        def energy(J0):
            return np.sum((J0 - target)**2)

        def grad_energy(detPhi1, dJ0, J0, J1):
            return self.regularizer.cauchy_navier_squared_inverse(2 * detPhi1[np.newaxis, ...] * dJ0 * (J0 - J1)[np.newaxis, ...])

        # set up variables
        self.T = T
        self.shape = input_.shape
        self.dim = input_.ndim
        self.opt = ()
        self.E_opt = None
        self.energy_threshold = 1e-3

        energies = []

        # define vector fields
        v = np.zeros((self.T, self.dim, *self.shape), dtype=np.double)
        dv = np.copy(v)

        # (12): iteration over k
        for k in range(K):

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
            for t in range(0, self.T):
                grad = grad_energy(detPhi1[t], dJ0[t], J0[t], J1[t])
                dv[t] = 2*v[t] - 1 / sigma**2 * grad

            # (10) calculate norm of the gradient, stop if small
            dv_norm = np.linalg.norm(dv)
            if dv_norm < 0.001:
                print(f"Gradient norm is {dv_norm} and therefore below threshold. Stopping ...")
                break

            # (11): calculate new energy
            E_regularizer = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(v[t])) for t in range(self.T)])
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
        length = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(v_hat[t])) for t in range(self.T)])

        if return_all:
            return self.opt + (length,)
        else:
            return Phi0[-1]

    def reparameterize(self, v):
        """
        implements step (2): Reparameterization of the velocity field to obtain a velocity field with constant speed.
        @param v: The velocity field.
        @return: Reparametrized velocity field, array.
        """
        length = np.sum([np.linalg.norm(self.regularizer.cauchy_navier(v[t])) for t in range(self.T)])
        for t in range(self.T):
            v[t] = length / self.T * v[t] / np.linalg.norm(self.regularizer.cauchy_navier(v[t]))
        return v

    def integrate_backward_flow(self, v):
        """
        implements step (3): Calculation of backward flows.
        @return: Flow, array.
        """
        # make identity grid
        x = grid.coordinate_grid(self.shape)

        # create flow
        Phi1 = np.zeros((self.T, self.dim, *self.shape), dtype=np.double)

        # Phi1_1 is the identity mapping
        Phi1[self.T - 1] = x

        for t in range(self.T-2, -1, -1):
            alpha = self.backwards_alpha(v[t], x)
            Phi1[t] = sampler.sample(Phi1[t + 1], x + alpha)

        return Phi1

    def backwards_alpha(self, v_t, x):
        """
        helper function for step (3): Calculation of backward flows.
        @param v_t: The velocity field at time `t`.
        @param x: Coordinates.
        @return: Alpha, array.
        """
        alpha = np.zeros(v_t.shape, dtype=np.double)
        for _ in range(5):
            alpha = sampler.sample(v_t, x + 0.5 * alpha)
        return alpha

    def integrate_forward_flow(self, v):
        """
        implements step (4): Calculation of forward flows.
        @param v: The velocity field.
        @return: Flow, array.
        """
        # make identity grid
        x = grid.coordinate_grid(self.shape)

        # create flow
        Phi0 = np.zeros((self.T, self.dim, *self.shape), dtype=np.double)

        # Phi0_0 is the identity mapping
        Phi0[0] = x

        for t in range(0, self.T-1):
            alpha = self.forward_alpha(v[t], x)
            Phi0[t+1] = sampler.sample(Phi0[t], x - alpha)

        return Phi0

    def forward_alpha(self, v_t, x):
        """
        helper function for step (4): Calculation of forward flows.
        @param v_t: The velocity field.
        @param x: Coordinates.
        @return: Alpha, array.
        """
        alpha = np.zeros(v_t.shape, dtype=np.double)
        for _ in range(5):
            alpha = sampler.sample(v_t, x - 0.5 * alpha)
        return alpha

    def push_forward(self, input_, Phi0):
        """
        implements step (5): Push forward image input_ along flow Phi0.
        @param input_: Image.
        @param Phi0: Flow.
        @return: Sequence of forward pushed images J0, array.
        """
        J0 = np.zeros((self.T,) + input_.shape, dtype=np.double)

        for t in range(0, self.T):
            J0[t] = sampler.sample(input_, Phi0[t])

        return J0

    def pull_back(self, target, Phi1):
        """
        implements step (6): Pull back image target along flow Phi1.
        @param target: Image.
        @param Phi1: Flow.
        @return: Sequence of back-pulled images J1, array.
        """
        J1 = np.zeros((self.T,) + target.shape, dtype=np.double)

        for t in range(self.T-1, -1, -1):
            J1[t] = sampler.sample(target, Phi1[t])

        return J1

    def image_grad(self, J0):
        """
        implements step (7): Calculate image gradients.
        @param J0: Sequence of forward pushed images J0.
        @return: Gradients of J0, array.
        """
        dJ0 = np.zeros((J0.shape[0], self.dim, *J0.shape[1:]), dtype=np.double)

        for t in range(self.T):
            dJ0[t] = finite_difference(J0[t])

        return dJ0

    def jacobian_determinant(self, Phi1):
        """
        implements step (8): Calculate Jacobian determinant of the transformation.
        @param Phi1: Sequence of transformations.
        @return: Sequence of determinants of J0, array.
        """
        detPhi1 = np.zeros((self.T, *self.shape), dtype=np.double)

        for t in range(self.T):
            if self.dim == 1:
                dx = finite_difference(Phi1[t, 0, ...])

                detPhi1[t] = dx[0, ...]
            elif self.dim == 2:
                # get gradient in x-direction
                dx = finite_difference(Phi1[t, 0, ...])
                # gradient in y-direction
                dy = finite_difference(Phi1[t, 1, ...])

                # calculate determinants
                detPhi1[t] = dx[0, ...] * dy[1, ...] - dx[1, ...] * dy[0, ...]
            elif self.dim == 3:
                # get gradient in x-direction
                dx = finite_difference(Phi1[t, 0, ...])
                # gradient in y-direction
                dy = finite_difference(Phi1[t, 1, ...])
                # gradient in z-direction
                dz = finite_difference(Phi1[t, 2, ...])

                # calculate determinants
                detPhi1[t] = (dx[0, ...] * dy[1, ...] * dz[2, ...]
                              + dy[0, ...] * dz[1, ...] * dx[2, ...]
                              + dz[0, ...] * dx[1, ...] * dy[2, ...]
                              - dx[2, ...] * dy[1, ...] * dz[0, ...]
                              - dy[2, ...] * dz[1, ...] * dx[0, ...]
                              - dz[2, ...] * dx[1, ...] * dy[0, ...])

        return detPhi1

    def is_injectivity_violated(self, detPhi1):
        """
        check injectivity: A function has a differentiable inverse iff the determinant of its jacobian is positive.
        @param detPhi1: Sequence of determinants of J0.
        @return: Truth value, bool.
        """
        return detPhi1.min() < 0

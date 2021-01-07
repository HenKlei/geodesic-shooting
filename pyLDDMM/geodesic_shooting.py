import numpy as np

from pyLDDMM.utils import sampler, grid
from pyLDDMM.utils.grad import finite_difference


class GeodesicShooting:
    """
    Geodesic shooting algorithm; Geodesic Shooting for Computational Anatomy.
    """
    def register(self, I0, problem, T=30, K=100, sigma=1, epsilon=0.01, return_all=False):
        """
        Registers two images.
        @param I0: image, ndarray.
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

        self.problem = problem

        I0 = I0.astype('double')
        if hasattr(self.problem, 'target'):
            target = self.problem.target.astype('double')
            assert I0.shape == target.shape

        # set up variables
        self.T = T
        self.shape = I0.shape
        self.dim = I0.ndim
        self.opt = (I0, None,)
        self.opt += ([],) * 2
        self.E_opt = None
        self.energy_threshold = 1e-3

        energies = []

        # define vector fields
        v0 = np.zeros((self.dim, *self.shape), dtype=np.double)
        v = np.zeros((self.T, self.dim, *self.shape), dtype=np.double)
        dv0 = np.copy(v0)

        # (12): iteration over k
        for k in range(K):

            # (1): Calculate new estimate of velocity
            v0 -= epsilon * dv0

            v = self.integrate_forward_vector_field(v0)

            # (4): calculate forward flows
            Phi0 = self.integrate_forward_flow(v)

            # (5): push-forward I0
            J0 = self.push_forward(I0, Phi0)

            # (7): Calculate image gradient
            dJ0 = self.image_grad(J0)

            dE1 = 1 / sigma**2 * self.problem.grad_energy(dJ0, J0)

            dv0 = - self.integrate_backward_adjoint_Jacobi_field_equations(dE1, v)

            # (10) calculate norm of the gradient, stop if small
            dv0_norm = np.linalg.norm(dv0)
            if dv0_norm < 0.001:
                print(f"Gradient norm is {dv0_norm} and therefore below threshold. Stopping ...")
                break

            # (11): calculate new energy
            E_regularizer = np.linalg.norm(self.problem.regularizer.L(v0))
            if hasattr(self.problem, 'target'):
                E_intensity = 1 / sigma**2 * self.problem.energy(J0)
            else:
                raise NotImplementedError
            E = E_regularizer + E_intensity

            if E < self.energy_threshold:
                self.E_opt = E
                self.opt = (J0, v0, energies, Phi0)
                print(f"Energy below threshold of {self.energy_threshold}. Stopping ...")
                break

            if self.E_opt is None or E < self.E_opt:
                self.E_opt = E
                self.opt = (J0, v0, energies, Phi0)

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
            length = np.linalg.norm(self.problem.regularizer.L(v_hat))
        else:
            length = 0.0

        if return_all:
            return self.opt + (length,)
        else:
            return Phi0

    def integrate_forward_flow(self, v):
        """
        implements step (4): Calculation of forward flows.
        @param v: The velocity field.
        @return: Flow, array.
        """
        # make identity grid
        x = grid.coordinate_grid(self.shape)

        # create flow
        Phi0 = np.zeros((self.dim, *self.shape), dtype=np.double)

        # Phi0_0 is the identity mapping
        Phi0 = x.astype(np.double)

        for t in range(0, self.T-1):
            alpha = self.forward_alpha(v[t], x)
            Phi0 = sampler.sample(Phi0, x - alpha)

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

    def push_forward(self, I0, Phi0):
        """
        implements step (5): Push forward image I0 along flow Phi0.
        @param I0: Image.
        @param Phi0: Flow.
        @return: Sequence of forward pushed images J0, array.
        """
        return sampler.sample(I0, Phi0)

    def image_grad(self, J0):
        """
        implements step (7): Calculate image gradients.
        @param J0: Sequence of forward pushed images J0.
        @return: Gradients of J0, array.
        """
        return finite_difference(J0)

    def integrate_forward_vector_field(self, v0):
        v = np.zeros((self.T, self.dim, *self.shape), dtype=np.double)
        v[0] = v0

        einsum_string = 'kl...,l...->k...'
        einsum_string_transpose = 'lk...,l...->k...'

        for t in range(0, self.T-2):
            mt = self.problem.regularizer.L(v[t])
            grad_mt = finite_difference(mt)[0:self.dim, ...]
            grad_vt = finite_difference(v[t])[0:self.dim, ...]
            div_vt = np.sum(np.array([grad_vt[d, d, ...] for d in range(self.dim)]), axis=0)
            rhs = np.einsum(einsum_string_transpose, grad_vt, mt) + np.einsum(einsum_string, grad_mt, v[t]) + mt * div_vt[np.newaxis, ...]
            v[t+1] = v[t] - self.problem.regularizer.K(rhs) / self.T

        return v

    def integrate_backward_adjoint_Jacobi_field_equations(self, dE1, v_seq):
        v_old = dE1
        v = dE1
        delta_v_old = np.zeros(v_old.shape, dtype=np.double)
        delta_v = delta_v_old

        einsum_string = 'kl...,l...->k...'
        einsum_string_transpose = 'lk...,l...->k...'

        for t in range(self.T-2, -1, -1):
            grad_v_seq = finite_difference(v_seq[t])[0:self.dim, ...]
            div_v_seq = np.sum(np.array([grad_v_seq[d, d, ...] for d in range(self.dim)]), axis=0)
            Lv = self.problem.regularizer.L(v_old)
            grad_Lv = finite_difference(Lv)[0:self.dim, ...]
            rhs_v = - self.problem.regularizer.K(np.einsum(einsum_string_transpose, grad_v_seq, Lv) + np.einsum(einsum_string, grad_Lv, v_seq[t]) + Lv * div_v_seq[np.newaxis, ...])
            v = v_old - rhs_v / self.T
            v_old = v

            grad_delta_v = finite_difference(delta_v)[0:self.dim, ...]
            div_delta_v = np.sum(np.array([grad_delta_v[d, d, ...] for d in range(self.dim)]), axis=0)
            Lv_seq = self.problem.regularizer.L(v_seq[t])
            grad_Lv_seq = finite_difference(Lv_seq)[0:self.dim, ...]
            rhs_delta_v = - v - (np.einsum(einsum_string, grad_v_seq, delta_v) - np.einsum(einsum_string, grad_delta_v, v_seq[t])) + self.problem.regularizer.K(np.einsum(einsum_string_transpose, grad_delta_v, Lv_seq) + np.einsum(einsum_string, grad_Lv_seq, delta_v) + Lv_seq * div_delta_v[np.newaxis, ...])
            delta_v = delta_v_old - rhs_delta_v / self.T
            delta_v_old = delta_v

        return delta_v

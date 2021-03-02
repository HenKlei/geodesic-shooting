import numpy as np

from pyLDDMM.regularizer import BiharmonicRegularizer


class PDEProblemStationary:

    def __init__(self, fom, alpha=1, gamma=1):
        """
        @param alpha: float, smoothness regularization. Higher values regularize stronger.
        @param gamma: float, norm penalty. Positive value to ensure injectivity of the regularizer.
        """
        assert alpha > 0
        assert gamma > 0

        self.fom = fom

        self.regularizer = BiharmonicRegularizer(alpha, gamma)

    def energy(self, J0, mu=None):
        J0 = self.fom.operator.source.from_numpy(J0.ravel())
        res = (self.fom.operator.apply(J0, mu) - self.fom.rhs.as_vector()).to_numpy()
        return np.sum(res**2)

    def grad_energy(self, detPhi1, dJ0, J0, mu=None):
        shape = J0.shape
        J0 = self.fom.operator.source.from_numpy(J0.ravel())
        d_res = self.fom.operator.jacobian(J0, mu).matrix
        res = (self.fom.operator.apply(J0, mu) - self.fom.rhs.as_vector()).to_numpy()
        return self.regularizer.K(2 * detPhi1[..., np.newaxis] * dJ0 * res.dot(d_res.toarray()).reshape(shape + (1,)))


class PDEProblemInstationary:

    def __init__(self, fom, u_old, solution_to_image, image_to_solution, dt=0.025, implicit_Euler=False, alpha=1, gamma=1):
        """
        @param alpha: float, smoothness regularization. Higher values regularize stronger.
        @param gamma: float, norm penalty. Positive value to ensure injectivity of the regularizer.
        """
        assert alpha > 0
        assert gamma > 0

        self.fom = fom
        self.u_old = u_old
        self.solution_to_image = solution_to_image
        self.image_to_solution = image_to_solution
        self.dt = dt
        self.implicit_Euler = implicit_Euler

        self.regularizer = BiharmonicRegularizer(alpha, gamma)

    def energy(self, J0, mu=None):
        if self.implicit_Euler:
            J0 = self.image_to_solution(J0)
            res = (self.image_to_solution(self.u_old) - self.dt * self.fom.operator.apply(J0, mu) - J0).to_numpy()
            return np.sum(res**2)
        else:
            J0 = self.image_to_solution(J0)
            res = (self.image_to_solution(self.u_old) - self.dt * self.fom.operator.apply(self.image_to_solution(self.u_old), mu) - J0).to_numpy()
            return np.sum(res**2)

    def grad_energy(self, detPhi1, dJ0, J0, mu=None):
        shape = J0.shape
        if self.implicit_Euler:
            J0 = self.image_to_solution(J0)
            d_res = - self.fom.operator.jacobian(J0, mu).matrix#.array()
            d_res = d_res - np.eye(d_res.shape[0])
            res = (self.image_to_solution(self.u_old) - self.dt * self.fom.operator.apply(J0, mu) - J0).to_numpy()
            return -self.regularizer.K(2 * detPhi1[..., np.newaxis] * dJ0 * self.solution_to_image(self.fom.operator.source.from_numpy(res.dot(d_res).squeeze())).reshape(shape + (1,)))
        else:
            J0 = self.image_to_solution(J0)
            res = (self.image_to_solution(self.u_old) - self.dt * self.fom.operator.apply(self.image_to_solution(self.u_old), mu) - J0)
            return -self.regularizer.K(2 * detPhi1[..., np.newaxis] * dJ0 * self.solution_to_image(res).reshape(shape + (1,)))

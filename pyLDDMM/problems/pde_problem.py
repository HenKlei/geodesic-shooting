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

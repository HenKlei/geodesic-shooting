import numpy as np

from pyLDDMM.regularizer import BiharmonicRegularizer


class ImageRegistrationProblem:

    def __init__(self, I1, alpha=1, gamma=1):
        """
        @param alpha: float, smoothness regularization. Higher values regularize stronger.
        @param gamma: float, norm penalty. Positive value to ensure injectivity of the regularizer.
        """
        assert alpha > 0
        assert gamma > 0

        self.I1 = I1
        self.regularizer = BiharmonicRegularizer(alpha, gamma)

    def energy(self, J0):
        return np.sum((J0 - self.I1)**2)

    def grad_energy(self, detPhi1, dJ0, J0, J1):
        return self.regularizer.K(2 * detPhi1[..., np.newaxis] * dJ0 * (J0 - J1)[..., np.newaxis])

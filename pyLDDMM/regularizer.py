import numpy as np
from scipy.ndimage import convolve

class BiharmonicReguarizer:
    def __init__(self, alpha=1, gamma=1):
        """
        Instantiates the Biharmonic regularizer.
        @param alpha: smoothness penalty. Positive number. The higher, the more smoothness will be enforced
        @param gamma: norm penalty. Positive value, so that the operator is non-singular
        """
        assert alpha > 0
        assert gamma > 0
        self.alpha = alpha
        self.gamma = gamma
        self.A = None

    def L(self, f):
        """
        The Cauchy-Navier operator (Equation 17).
        @param f: an array representing function f
        @return: g = L(f), array
        """
        assert f.ndim in [2, 3]

        if f.ndim == 2:
            w = np.array([1., -2., 1.])
        elif f.ndim == 3:
            w = np.array([[0., 1., 0.],
                          [1., -4., 1.],
                          [0., 1., 0.]])

        dff = np.stack([convolve(f[..., d], w) for d in range(f.shape[-1])], axis=-1)

        return - self.alpha * dff + self.gamma * f

    def K(self, g):
        """
        The K = (LL)^-1 operator.
        @param g: an array representing function g
        @return: f = K(g) = (LL)^-1 (g), array
        """
        if self.A is None or self.A.shape != g.shape[:-1]:
            # A is not chached. compute A.
            self.A = self.compute_A(g.shape)

        # transform to fourier domain
        G = self.fftn(g)

        # perform operation in fourier space
        F = G / self.A**2

        # transform back to normal domain
        f = self.ifftn(F)
        return f

    def compute_A(self, shape):
        """
        Computes the A(k) operator.
        @param shape: shape of the input image
        @return: A(k)
        """
        dim = shape[-1]
        shape = shape[:-1]
        A = np.zeros(shape, dtype=np.double)

        for i in np.ndindex(shape):
            for d in range(dim):
                A[i] += 2 * self.alpha * (1 - np.cos(2 * np.pi * i[d] / shape[d]))

        A += self.gamma

        # expand dims to match G
        A = np.stack([A,]*dim, axis=-1)
        return A

    def fftn(self, a):
        """
        Performs n-d FFT along the first n axes of a (n+1)-d array.
        """
        C = a.shape[-1]
        A = np.zeros(a.shape, dtype=np.complex128)
        for c in range(C):
            A[..., c] = np.fft.fftn(a[..., c])
        return A

    def ifftn(self, A):
        """
        Performs n-d iFFT along the first n axes of a (n+1)-d array.
        """
        C = A.shape[-1]
        a = np.zeros(A.shape, dtype=np.complex128)
        for c in range(C):
            a[..., c] = np.fft.ifftn(A[..., c])
        return np.real(a)

if __name__ == '__main__':
    v = np.zeros((5, 5, 2))
    v[2, 2, 0] = 1

    reg = BiharmonicReguarizer(alpha=1, gamma=1)

    print(reg.K(v)[..., 0])

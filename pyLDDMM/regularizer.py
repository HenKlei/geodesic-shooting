import numpy as np
from scipy.ndimage import convolve


class BiharmonicRegularizer:
    def __init__(self, alpha=1, gamma=1):
        """
        Instantiates the Biharmonic regularizer.
        @param alpha: Smoothness penalty. Positive number. The higher, the more smoothness will be enforced.
        @param gamma: Norm penalty. Positive value, so that the operator is non-singular.
        """
        assert alpha > 0
        assert gamma > 0
        self.alpha = alpha
        self.gamma = gamma
        self.A = None

    def L(self, f):
        """
        Application of the Cauchy-Navier operator `L` to a function `f` (Equation 17).
        @param f: An array representing the function `f`.
        @return: `g=L(f)`, array.
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
        Application of the compact and self-adjoint smoothing operator `K=(LL)^-1` to a function `g`.
        @param g: An array representing the function `g`.
        @return: `f=K(g)=(LL)^-1(g)`, array.
        """
        if self.A is None or self.A.shape != g.shape[:-1]:
            # A is not chached. compute A.
            self.A = self.compute_A(g.shape)

        # transform to fourier domain
        G = self.fftn(g)

        # perform operation in fourier space
        F = G / self.A**2

        # transform back to normal domain
        return self.ifftn(F)

    def compute_A(self, shape):
        """
        Computes the operator `A(k)`.
        @param shape: Shape of the input image.
        @return: `A(k)`, array.
        """
        dim = shape[-1]
        shape = shape[:-1]
        A = np.zeros(shape, dtype=np.double)

        for i in np.ndindex(shape):
            for d in range(dim):
                A[i] += 2 * self.alpha * (1 - np.cos(2 * np.pi * i[d] / shape[d]))

        A += self.gamma

        # expand dims to match G
        return np.stack([A,]*dim, axis=-1)

    def fftn(self, a):
        """
        Performs `n`-dimensional FFT along the first `n` axes of an `n+1`-dimensional array.
        """
        C = a.shape[-1]
        A = np.zeros(a.shape, dtype=np.complex128)
        for c in range(C):
            A[..., c] = np.fft.fftn(a[..., c])
        return A

    def ifftn(self, A):
        """
        Performs the `n`-dimensional inverse FFT (iFFT) along the first `n` axes of an `n+1`-dimensional array.
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

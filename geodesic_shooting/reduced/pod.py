import numpy as np


def pod(A, modes=10):
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, :min(modes, U.shape[1])]

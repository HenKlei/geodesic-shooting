import numpy as np


def pod(modes, num_modes=10, product_operator=None, return_singular_values=False, shift=None):
    assert isinstance(num_modes, int) and num_modes > 0
    type_input = type(modes[0])

    if shift is None:
        shift = type_input(spatial_shape=modes[0].spatial_shape)

    if product_operator:
        B = np.stack([a.flatten() for a in modes])
        B_tilde = np.stack([product_operator(a).flatten() for a in modes])
        C = B.dot(B_tilde.T)
        S, V = np.linalg.eig(C)
        selected_modes = min(num_modes, V.shape[0])
        all_singular_values = np.real(S)
        S = np.sqrt(S[:selected_modes])
        V = V.T
        V = B.T.dot((V[:selected_modes] / S[:, np.newaxis]).T)
        singular_vectors = np.real(V).T
        singular_vectors = [type_input(data=u.reshape(modes[0].full_shape)) + shift for u in singular_vectors]
        if return_singular_values:
            return singular_vectors, all_singular_values
        else:
            return singular_vectors
    else:  # assuming L2-scalar product as default if no `product_operator` is provided
        B = np.stack([a.flatten() for a in modes]).T
        assert B.ndim == 2
        U, all_singular_values, _ = np.linalg.svd(B, full_matrices=False)
        singular_vectors = U[:, :min(num_modes, U.shape[1])].T
        singular_vectors = [type_input(data=u.reshape(modes[0].full_shape)) + shift for u in singular_vectors]
        if return_singular_values:
            return singular_vectors, all_singular_values
        else:
            return singular_vectors

import numpy as np


def pod(modes, num_modes=10, product_operator=None, return_singular_values='all', shift=None):
    """Proper orthogonal decomposition of the given modes.

    This function computes the POD of the given modes. It is possible to pass a
    product to the method which is used to compute the gramian matrix.

    Parameters
    ----------
    modes
        Modes to compute the proper orthogonal decomposition of.
    num_modes
        Number of modes to return.
    product_operator
        Operator to compute the inner product for the gramian matrix.
    return_singular_values
        Determines whether to return (all) singular values.
        If `'all'`, all singular values are returned. If `True`, only the
        singular values to the POD modes are returned. If `False`, no
        singular values are returned.
    shift
        Shift added to the final singular vectors before returning them.

    Returns
    -------
    Either only a list of the POD modes or in addition the singular values.
    See also the description of `return_singular_values`.
    """
    assert isinstance(num_modes, int) and num_modes > 0
    type_input = type(modes[0])

    if shift is None:
        if hasattr(modes[0], "spatial_shape"):
            shift = type_input(spatial_shape=modes[0].spatial_shape)
        else:
            shift = np.zeros(modes[0].shape)

    if product_operator:
        B = np.stack([a.flatten() for a in modes])
        B_tilde = np.stack([product_operator(a).flatten() for a in modes])
        C = B.dot(B_tilde.T)
        S, V = np.linalg.eig(C)
        selected_modes = min(num_modes, V.shape[0])
        S[(S <= 0.) | np.isclose(S, 0.)] = 0.
        all_singular_values = np.sqrt(S)
        idx = all_singular_values.argsort()[::-1]
        all_singular_values = all_singular_values[idx]
        V = V[idx]
        S = all_singular_values[:selected_modes]
        V = V.T
        S_pos = S.copy()
        S_pos[np.isclose(S_pos, 0.)] = 1.
        V = B.T.dot((V[:selected_modes] / S_pos[:, np.newaxis]).T)
        singular_vectors = np.real(V).T
        singular_vectors = [type_input(data=u.reshape(modes[0].full_shape)) + shift for u in singular_vectors]
        u_norms = [u.get_norm(product_operator=product_operator) for u in singular_vectors]
        singular_vectors = [u / norm if not np.isclose(norm, 0.) else u for u, norm in zip(singular_vectors, u_norms)]
        if return_singular_values == 'all':
            return singular_vectors, np.real(all_singular_values)
        elif return_singular_values:
            return singular_vectors, np.real(S)
        else:
            return singular_vectors
    else:  # assuming L2-scalar product as default if no `product_operator` is provided
        B = np.stack([a.flatten() for a in modes]).T
        assert B.ndim == 2
        U, all_singular_values, _ = np.linalg.svd(B, full_matrices=False)
        selected_modes = min(num_modes, U.shape[1])
        singular_vectors = U[:, :selected_modes].T
        if hasattr(modes[0], "full_shape"):
            singular_vectors = [type_input(data=u.reshape(modes[0].full_shape)) + shift for u in singular_vectors]
        else:
            singular_vectors = [u.reshape(modes[0].shape) + shift for u in singular_vectors]
        if return_singular_values == 'all':
            return singular_vectors, all_singular_values
        elif return_singular_values:
            return singular_vectors, all_singular_values[:selected_modes]
        else:
            return singular_vectors

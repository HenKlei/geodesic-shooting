import numpy as np
import scipy.sparse as sp


def prod(val):
    res = 1
    for ele in val:
        res *= ele
    return res


def laplacian_matrix(input_shape):
    size = prod(input_shape)
    mat = sp.csr_matrix((size, size))
    for dim in range(len(input_shape)):
        mat += recursive_kronecker_product(dim, input_shape)
    return mat


def recursive_kronecker_product(dim, input_shape, i=0):
    if len(input_shape) == 1:
        diag = np.ones(input_shape[dim])
        laplacian = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1], input_shape[dim], input_shape[dim])
        return laplacian
    if i == len(input_shape) - 2:
        if i == dim:
            diag = np.ones(input_shape[dim])
            laplacian = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1], input_shape[dim], input_shape[dim])
            return sp.kron(laplacian, sp.eye(input_shape[i+1]))
        elif dim == len(input_shape) - 1:
            diag = np.ones(input_shape[dim])
            laplacian = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1], input_shape[dim], input_shape[dim])
            return sp.kron(sp.eye(input_shape[i]), laplacian)
        else:
            return sp.kron(sp.eye(input_shape[i]), sp.eye(input_shape[i+1]))
    else:
        if i == dim:
            diag = np.ones(input_shape[dim])
            laplacian = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1], input_shape[dim], input_shape[dim])
            return sp.kron(laplacian, recursive_kronecker_product(dim, input_shape, i+1))
        else:
            return sp.kron(sp.eye(input_shape[i]), recursive_kronecker_product(dim, input_shape, i+1))


if __name__ == '__main__':
    shape = (10, )
    print(laplacian_matrix(shape).todense())
    print(laplacian_matrix(shape).shape)

    shape = (10, 15)
    print(laplacian_matrix(shape).todense())
    print(laplacian_matrix(shape).shape)

    shape = (10, 15, 5)
    print(laplacian_matrix(shape).todense())
    print(laplacian_matrix(shape).shape)

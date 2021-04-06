import numpy as np

from geodesic_shooting.utils.regularizer import BiharmonicRegularizer


def test_regularizer():
    v = np.zeros((2, 5, 5))
    v[0, 2, 2] = 1

    regularizer = BiharmonicRegularizer(alpha=1, exponent=1)

    print(regularizer.cauchy_navier_squared_inverse(v)[0])

    image = np.zeros((2, 5, 5))
    image[0, 2, 2] = 1

    regularizer = BiharmonicRegularizer(alpha=1, exponent=1)
    regularizer.init_matrices(shape=image[0].shape)

    assert (regularizer.cauchy_navier(image) == np.stack([regularizer.cauchy_navier_matrix.dot(
        elem.flatten()).reshape(elem.shape) for elem in image])).all()

    exact = np.zeros((2, 5, 5))
    exact[0, 2, 2] = 5
    exact[0, 2, 1] = -1
    exact[0, 1, 2] = -1
    exact[0, 2, 3] = -1
    exact[0, 3, 2] = -1
    assert (regularizer.cauchy_navier(image) == exact).all()
    square_matrix = regularizer.cauchy_navier_matrix.dot(regularizer.cauchy_navier_matrix)

    regularizer = BiharmonicRegularizer(alpha=1, exponent=2)
    regularizer.init_matrices(shape=image.shape[1:])
    print(regularizer.cauchy_navier_squared_inverse(image)[0])
    print(regularizer.cauchy_navier_inverse_matrix.dot(image[0].flatten()).reshape(image[0].shape))

    assert np.count_nonzero(regularizer.cauchy_navier_matrix != square_matrix) == 0

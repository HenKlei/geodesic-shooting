import numpy as np

import geodesic_shooting
from geodesic_shooting.utils.kernels import GaussianKernel


def test_kernel_gramian():
    positions = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    num_elements, dim = positions.shape
    size = num_elements * dim
    kernel = GaussianKernel()
    mat = []
    pos = positions.reshape((size // dim, dim))

    for i in range(size // dim):
        mat_row = []
        for j in range(size // dim):
            mat_row.append(kernel(pos[i], pos[j]))
        mat.append(mat_row)

    block_mat = np.block(mat)
    assert block_mat.shape == (size, size)

    block_mat_2 = kernel.apply_vectorized(positions, positions, dim)
    assert block_mat_2.shape == (size, size)

    assert np.allclose(block_mat, block_mat_2)


def test_landmark_matching():
    def compute_average_distance(target_landmarks, registered_landmarks):
        dist = 0.
        for x, y in zip(registered_landmarks, target_landmarks):
            dist += np.linalg.norm(x - y)
        dist /= registered_landmarks.shape[0]
        return dist

    input_landmarks = np.array([[5., 3.]])
    target_landmarks = np.array([[6., 2.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.001, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-5

    input_landmarks = np.array([[5., 3.]])
    target_landmarks = np.array([[6., 2.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.001, return_all=True,
                         landmarks_labeled=True, optimization_method='newton')
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-5

    input_landmarks = np.array([[5., 3.], [1., 1.]])
    target_landmarks = np.array([[6., 2.], [0., 0.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.001, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-5

    input_landmarks = np.array([[5., 3.], [1., 1.]])
    target_landmarks = np.array([[6., 2.], [0., 0.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.001, return_all=True,
                         landmarks_labeled=False)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-3

    input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])

    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={'sigma': .5})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.01, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-2

    input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])

    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={'sigma': .5})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.01, return_all=True,
                         landmarks_labeled=True, optimization_method='newton')
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-2

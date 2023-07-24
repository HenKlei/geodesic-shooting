import numpy as np

import geodesic_shooting


def test_differential_computation():
    input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])

    gs = geodesic_shooting.LandmarkShooting()

    w = np.zeros(gs.dim)
    for i in range(gs.dim):
        ei = np.zeros(gs.dim)
        ei[i] = 1.
        w[i] = ((gs.DK(position) @ ei) @ momentum).dot(momentum)
    assert np.allclose(w, (momentum.T @ gs.DK(position) @ momentum))


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
    result = gs.register(input_landmarks, target_landmarks, sigma=0.05, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-5

    input_landmarks = np.array([[5., 3.], [1., 1.]])
    target_landmarks = np.array([[6., 2.], [0., 0.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.05, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-5

    input_landmarks = np.array([[5., 3.], [1., 1.]])
    target_landmarks = np.array([[6., 2.], [0., 0.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.05, return_all=True,
                         landmarks_labeled=False)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-3

    input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])

    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={'sigma': .5})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.1, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']
    dist = compute_average_distance(target_landmarks, registered_landmarks)
    assert dist < 1e-2

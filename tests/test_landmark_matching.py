import numpy as np

import geodesic_shooting


def test_landmark_matching():
    input_landmarks = np.array([[5., 3.]])
    target_landmarks = np.array([[6., 2.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.05, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']

    assert np.linalg.norm(registered_landmarks - target_landmarks) < 1e-5

    input_landmarks = np.array([[5., 3.], [1., 1.]])
    target_landmarks = np.array([[6., 2.], [0., 0.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.05, return_all=True,
                         landmarks_labeled=True)
    registered_landmarks = result['registered_landmarks']

    assert np.linalg.norm(registered_landmarks - target_landmarks) < 1e-5

    input_landmarks = np.array([[5., 3.], [1., 1.]])
    target_landmarks = np.array([[6., 2.], [0., 0.]])

    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.05, return_all=True,
                         landmarks_labeled=False,)
    registered_landmarks = result['registered_landmarks']

    assert np.linalg.norm(registered_landmarks - target_landmarks) < 1e-3

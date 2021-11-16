import numpy as np

import geodesic_shooting


if __name__ == "__main__":
    # define landmark positions
    input_landmarks = np.array([[1.], [2.], [9.]])
    target_landmarks = np.array([[3.], [4.5], [8.]])

    # perform the registration using landmark shooting algorithm
    gs = geodesic_shooting.LandmarkShooting()
    result = gs.register(input_landmarks, target_landmarks, sigma=0.05, return_all=True)
    registered_landmarks = result['registered_landmarks']

    print(f"Input: {input_landmarks}")
    print(f"Target: {target_landmarks}")
    print(f"Result: {registered_landmarks}")
    rel_error = (np.linalg.norm(target_landmarks - registered_landmarks)
                 / np.linalg.norm(target_landmarks))
    print(f"Relative norm of difference: {rel_error}")

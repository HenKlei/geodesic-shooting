import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting


if __name__ == "__main__":
    # define greyscale images
    input_landmarks = np.array([[5., 3.], [4., 2.], [1., 0.], [2., 3.]])
    print(input_landmarks.shape)
    print(input_landmarks)
    target_landmarks = np.array([[6., 2.], [5., 1.], [1., -1.], [2.5, 2.]])
    print(target_landmarks.shape)
    print(target_landmarks)

    # perform the registration
    gs = geodesic_shooting.LandmarkShooting()
    final_momenta, final_positions = gs.register(input_landmarks, target_landmarks)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    for landmark in range(input_landmarks.shape[0]):
        axis.scatter(input_landmarks[landmark][0], input_landmarks[landmark][1],
                     s=100, color=f'C{landmark}')
        axis.scatter(input_landmarks[landmark][0], input_landmarks[landmark][1],
                     s=10, color='black')
        axis.scatter(target_landmarks[landmark][0], target_landmarks[landmark][1],
                     s=100, color=f'C{landmark}')
        axis.scatter(target_landmarks[landmark][0], target_landmarks[landmark][1],
                     s=10, color='white')
        axis.scatter(final_positions[landmark][0], final_positions[landmark][1],
                     s=100, color=f'C{landmark}')
    plt.show()

    print(f"Input: {input_landmarks}")
    print(f"Target: {target_landmarks}")
    rel_error = (np.linalg.norm(target_landmarks - final_positions)
                 / np.linalg.norm(target_landmarks))
    print(f"Relative norm of difference: {rel_error}")

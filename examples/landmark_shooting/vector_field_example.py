import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import geodesic_shooting


if __name__ == "__main__":
    input_landmarks = np.array([[-1., 0.], [-0.5, 1.], [1., 0.]])

    gs = geodesic_shooting.LandmarkShooting(dim=2, num_landmarks=input_landmarks.shape[0])

    initial_momenta = np.array([[1./np.sqrt(2.), 1./np.sqrt(2.)],
                                [-2./np.sqrt(5.), 1./np.sqrt(5.)],
                                [-2./np.sqrt(5.), -1./np.sqrt(5.)]])

    vector_field = gs.construct_vector_field(input_landmarks, initial_momenta)

    min_x = -2.
    max_x = 2.
    min_y = -2.
    max_y = 2.

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    N = 30
    xs = np.array([[x for x in np.linspace(min_x, max_x, N)] for _ in np.linspace(min_y, max_y, N)])
    ys = np.array([[y for _ in np.linspace(min_x, max_x, N)] for y in np.linspace(min_y, max_y, N)])
    vector_field_x = np.array([[vector_field(np.array([x, y]))[0]
                               for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])
    vector_field_y = np.array([[vector_field(np.array([x, y]))[1]
                               for x in np.linspace(min_x, max_x, N)]
                              for y in np.linspace(min_y, max_y, N)])

    axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=2., angles='xy', scale_units='xy')

    for i, (landmark, momentum) in enumerate(zip(input_landmarks, initial_momenta)):
        axis.scatter(landmark[0], landmark[1], s=100, color=f'C{i}')
        axis.arrow(landmark[0], landmark[1], momentum[0], momentum[1],
                   head_width=0.05, color=f'C{i}')

    plt.show()

    momenta, positions = gs.integrate_forward_Hamiltonian(initial_momenta.flatten(),
                                                          input_landmarks.flatten())

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.set_aspect('equal')

    N = 30
    xs = np.array([[x for x in np.linspace(-2., 2., N)] for _ in np.linspace(-2., 2., N)])
    ys = np.array([[y for _ in np.linspace(-2., 2., N)] for y in np.linspace(-2., 2., N)])
    vector_field_x = np.array([[vector_field(np.array([x, y]))[0] for x in np.linspace(-2., 2., N)]
                              for y in np.linspace(-2., 2., N)])
    vector_field_y = np.array([[vector_field(np.array([x, y]))[1] for x in np.linspace(-2., 2., N)]
                              for y in np.linspace(-2., 2., N)])

    axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=2., angles='xy', scale_units='xy')

    for pos in positions:
        pos = pos.reshape(input_landmarks.shape)
        for i, landmark in enumerate(pos):
            axis.scatter(landmark[0], landmark[1], s=100, color=f'C{i}')

    plt.show()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    def plot_positions_and_velocity_field(momenta, positions):

        axis.set_aspect('equal')

        vector_field = gs.construct_vector_field(positions[-1], momenta[-1])

        N = 30
        xs = np.array([[x for x in np.linspace(min_x, max_x, N)]
                       for _ in np.linspace(min_y, max_y, N)])
        ys = np.array([[y for _ in np.linspace(min_x, max_x, N)]
                       for y in np.linspace(min_y, max_y, N)])
        vector_field_x = np.array([[vector_field(np.array([x, y]))[0]
                                   for x in np.linspace(min_x, max_x, N)]
                                  for y in np.linspace(min_y, max_y, N)])
        vector_field_y = np.array([[vector_field(np.array([x, y]))[1]
                                   for x in np.linspace(min_x, max_x, N)]
                                  for y in np.linspace(min_y, max_y, N)])

        axis.quiver(xs, ys, vector_field_x, vector_field_y, scale=2., angles='xy', scale_units='xy')

        for j, pos in enumerate(positions):
            for i, landmark in enumerate(pos):
                size = 100 if j == len(positions) - 1 else 10
                axis.scatter(landmark[0], landmark[1],
                             s=size, color=f'C{i}')

        return fig

    def animate(i):
        axis.clear()
        pos = positions[:i+1].reshape((i+1, *input_landmarks.shape))
        mom = momenta[:i+1].reshape((i+1, *initial_momenta.shape))
        plot_positions_and_velocity_field(mom, pos)

    ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=1000)
    plt.show()

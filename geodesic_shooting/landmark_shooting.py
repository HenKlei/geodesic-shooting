import time
import numpy as np

from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.kernels import GaussianKernel


class LandmarkShooting:
    """Class that implements large deformation metric mappings via geodesic shooting.

    Based on:
    Geodesic Shooting for Computational Anatomy.
    Miller, Trouvé, Younes, 2006
    """
    def __init__(self, kernel=GaussianKernel, dim=2, num_landmarks=1, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        dim
            Dimension of the input and target images (set automatically when calling `register`).
        """
        self.time_steps = 30
        self.dt = 1. / self.time_steps

        self.dim = dim
        self.size = dim * num_landmarks

        self.kernel = kernel()

        self.logger = getLogger('landmark_shooting', level=log_level)

    def register(self, input_landmarks, target_landmarks, time_steps=30, iterations=10000,
                 delta=0.0001, sigma=0.25, early_stopping=10, initial_momenta=None, return_all=False):
        """Performs actual registration according to LDDMM algorithm with time-varying velocity
           fields that are chosen via geodesics.

        Parameters
        ----------
        time_steps
            Number of discrete time steps to perform.
        """
        assert isinstance(time_steps, int)
        self.time_steps = time_steps
        self.dt = 1. / self.time_steps
        assert early_stopping is None or (isinstance(early_stopping, int) and early_stopping > 0)
        assert input_landmarks.ndim == 2
        assert input_landmarks.shape == target_landmarks.shape
        self.dim = input_landmarks.shape[1]
        self.size = input_landmarks.shape[0] * input_landmarks.shape[1]

        # define initial momenta
        if initial_momenta is None:
            initial_momenta = (target_landmarks - input_landmarks)
        assert initial_momenta.shape == input_landmarks.shape

        momenta = initial_momenta.flatten()
        positions = input_landmarks.flatten()

        def compute_matching_function(positions):
            return np.linalg.norm(positions - target_landmarks.flatten())**2

        def compute_gradient_matching_function(positions):
            return (positions - target_landmarks.flatten()) / sigma**2

        def set_opt(opt, energy, energy_regularizer, energy_intensity, energy_intensity_unscaled,
                    transformed_landmarks, initial_momenta):
            opt['energy'] = energy
            opt['energy_regularizer'] = energy_regularizer
            opt['energy_intensity'] = energy_intensity
            opt['energy_intensity_unscaled'] = energy_intensity_unscaled
            opt['transformed_landmarks'] = transformed_landmarks
            opt['initial_momenta'] = initial_momenta
            return opt

        opt = set_opt({}, None, None, None, None, input_landmarks, initial_momenta)

        k = 0
        reason_registration_ended = 'reached maximum number of iterations'

        start_time = time.perf_counter()

        with self.logger.block("Perform image matching via geodesic shooting ..."):
            try:
                while k < iterations:
                    momenta_time_dependent, positions_time_dependent = self.integrate_forward_Hamiltonian(
                        momenta, input_landmarks.flatten())
                    positions = positions_time_dependent[-1]

                    d_momenta_1 = self.integrate_forward_variational_Hamiltonian(momenta_time_dependent,
                                                                                 positions_time_dependent)
                    grad_g = compute_gradient_matching_function(positions)
                    grad = (self.K(input_landmarks.flatten()) @ momenta + d_momenta_1.T @ grad_g)
                    momenta = momenta - delta * grad

                    energy_regularizer = self.compute_Hamiltonian(momenta, positions)
                    energy_intensity_unscaled = compute_matching_function(positions)
                    energy_intensity = 1. / (2. * sigma**2) * energy_intensity_unscaled
                    energy = energy_regularizer + energy_intensity

                    # update optimal energy if necessary
                    if opt['energy'] is None or energy < opt['energy']:
                        opt = set_opt(opt, energy, energy_regularizer, energy_intensity,
                                      energy_intensity_unscaled, positions.reshape((-1, self.dim)),
                                      momenta.reshape((-1, self.dim)))
                        energy_not_decreasing = 0
                    else:
                        energy_not_decreasing += 1

                    if early_stopping is not None and energy_not_decreasing >= early_stopping:
                        self.logger.info(f"Energy did not decrease for {early_stopping} "
                                         "iterations. Early stopping ...")
                        reason_registration_ended = 'early stopping due to non-decreasing energy'
                        break

                    k += 1

                    # output of current iteration and energies
                    self.logger.info(f"iter: {k:3d}, "
                                     f"energy: {energy:4.2f}, "
                                     f"L2 (w/o scale): {energy_intensity_unscaled:4.4f}, "
                                     f"reg: {energy_regularizer:4.2f}")
            except KeyboardInterrupt:
                self.logger.warning("Aborting registration ...")
                reason_registration_ended = 'manual abort'

        elapsed_time = int(time.perf_counter() - start_time)

        self.logger.info("Finished registration ...")

        opt['time'] = elapsed_time
        opt['reason_registration_ended'] = reason_registration_ended

        if return_all:
            return opt
        return momenta.reshape((-1, self.dim))

    def compute_Hamiltonian(self, momenta, positions):
        return 0.5 * momenta.T @ self.K(positions) @ momenta

    def integrate_forward_Hamiltonian(self, initial_momenta, initial_positions):
        assert initial_momenta.shape == (self.size,)
        assert initial_positions.shape == (self.size,)
        momenta = np.zeros((self.time_steps, self.size))
        positions = np.zeros((self.time_steps, self.size))
        momenta[0] = initial_momenta
        positions[0] = initial_positions

        for t in range(self.time_steps-1):
            momenta[t+1] = momenta[t] - self.dt * 0.5 * (momenta[t].T @ self.DK(positions[t])
                                                         @ momenta[t])
            positions[t+1] = positions[t] + self.dt * self.K(positions[t]) @ momenta[t]

        return momenta, positions

    def K(self, positions):
        assert positions.shape == (self.size, )
        mat = []
        pos = positions.reshape((self.size // self.dim, self.dim))
        for i in range(self.size // self.dim):
            mat_row = []
            for j in range(self.size // self.dim):
                mat_row.append(self.kernel(pos[i], pos[j]))
            mat.append(mat_row)
        block_mat = np.block(mat)
        assert block_mat.shape == (self.size, self.size)
        return block_mat

    def DK(self, positions):
        mat = np.zeros((self.size, self.size, self.size))
        for i in range(self.size):
            modi = i % self.dim
            for j in range(self.size):
                modj = j % self.dim
                for k in range(self.size):
                    if i == k:
                        mat[i][j][k] += self.kernel.derivative_1(positions[i-modi:i+self.dim-modi],
                                                                 positions[j-modj:j+self.dim-modj],
                                                                 modi)  # modk
                    if j == k:
                        mat[i][j][k] += self.kernel.derivative_2(positions[i-modi:i+self.dim-modi],
                                                                 positions[j-modj:j+self.dim-modj],
                                                                 modj)  # modk
        return mat

    def construct_vector_field(self, positions, momenta):
        assert positions.ndim == 2
        assert positions.shape == momenta.shape

        def vector_field(x):
            result = np.zeros(positions.shape[1])
            for q, p in zip(positions, momenta):
                result += self.kernel(x, q) @ p
            return result

        return vector_field

    def integrate_forward_variational_Hamiltonian(self, momenta, positions):
        assert positions.shape == (self.time_steps, self.size)
        assert momenta.shape == (self.time_steps, self.size)

        d_positions = np.zeros((self.time_steps, self.size, self.size))
        d_momenta = np.zeros((self.time_steps, self.size, self.size))
        d_momenta[0] = np.eye(self.size)

        for t in range(self.time_steps-1):
            d_positions[t+1] = d_positions[t] + self.dt * (self.DK(positions[t]) @ d_positions[t]
                                                           @ momenta[t]
                                                           + self.K(positions[t]) @ d_momenta[t])
            d_momenta[t+1] = d_momenta[t] - self.dt * (
                d_momenta[t] @ self.DK(positions[t]) @ positions[t]
                + positions[t] @ self.DK(positions[t]) @ d_momenta[t])
#                + positions[t] @ @ positions[t]  # maybe a term is missing here...

        return d_momenta[-1]

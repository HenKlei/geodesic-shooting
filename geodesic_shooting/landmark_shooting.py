import time
import numpy as np

from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.kernels import GaussianKernel
from geodesic_shooting.utils.visualization import construct_vector_field
from geodesic_shooting.utils.optim import GradientDescentOptimizer, ArmijoLineSearch, PatientStepsizeController


class LandmarkShooting:
    """Class that implements geodesic shooting for landmark matching.

    Based on:
    Geodesic Shooting and Diffeomorphic Matching Via Textured Meshes.
    Allassonnière, Trouvé, Younes, 2005
    """
    def __init__(self, kernel=GaussianKernel, dim=2, num_landmarks=1, log_level='INFO', kwargs_kernel={}):
        """Constructor.

        Parameters
        ----------
        kernel
            Kernel to use for extending the velocity fields to the whole domain.
        dim
            Dimension of the landmarks (set automatically when calling `register`).
        num_landmarks
            Number of landmarks to register (set automatically when calling `register`).
        log_level
            Verbosity of the logger.
        """
        self.time_steps = 30
        self.dt = 1. / self.time_steps

        self.dim = dim
        self.size = dim * num_landmarks

        self.kernel = kernel(**kwargs_kernel)

        self.logger = getLogger('landmark_shooting', level=log_level)

    def register(self, input_landmarks, target_landmarks, time_steps=30, sigma=1.,
                 OptimizationAlgorithm=GradientDescentOptimizer, iterations=1000, early_stopping=10,
                 initial_momenta=None, LineSearchAlgorithm=ArmijoLineSearch,
                 parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1e-3,
                                         'max_num_search_steps': 10},
                 StepsizeControlAlgorithm=PatientStepsizeController,
                 energy_threshold=1e-6, gradient_norm_threshold=1e-6,
                 return_all=False):
        """Performs actual registration according to geodesic shooting algorithm for landmarks using
           a Hamiltonian setting.

        Parameters
        ----------
        input_landmarks
            Initial positions of the landmarks.
        target_landmarks
            Target positions of the landmarks.
        time_steps
            Number of discrete time steps to perform.
        sigma
            Weight for the similarity measurement (L2 difference of the target and the registered
            landmarks); the smaller sigma, the larger the influence of the L2 loss.
        OptimizationAlgorithm
            Algorithm to use for optimization during registration. Should be a class and not an
            instance. The class should derive from `BaseOptimizer`.
        iterations
            Number of iterations of the optimizer to perform.
                epsilon
            Learning rate, i.e. step size of the optimizer.
        early_stopping
            Number of iterations with non-decreasing energy after which to stop registration.
            If `None`, no early stopping is used.
        initial_momenta
            Used as initial guess for the initial momenta (will agree with the direction pointing
            from the input landmarks to the target landmarks if None is passed).
        LineSearchAlgorithm
            Algorithm to use as line search method during optimization. Should be a class and not
            an instance. The class should derive from `BaseLineSearch`.
        parameters_line_search
            Additional parameters for the line search algorithm
            (e.g. minimal and maximal stepsize, ...).
        StepsizeControlAlgorithm
            Algorithm to use for adjusting the minimal and maximal stepsize (or other parameters
            of the line search algorithm that are prescribed in `parameters_line_search`).
            The class should derive from `BaseStepsizeController.
        energy_threshold
            If the energy drops below this threshold, the registration is stopped.
        gradient_norm_threshold
            If the norm of the gradient drops below this threshold, the registration is stopped.
        return_all
            Determines whether or not to return all information or only the initial momenta
            that led to the best registration result.

        Returns
        -------
        Either the best initial momenta (if return_all is False) or a dictionary consisting of the
        transformed/registered landmarks, the initial momenta, the energies and the time evolutions
        of momenta and positions (if return_all is True).
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
        else:
            initial_momenta = np.reshape(initial_momenta, input_landmarks.shape)
        assert initial_momenta.shape == input_landmarks.shape

        initial_momenta = initial_momenta.flatten()
        initial_positions = input_landmarks.flatten()

        def compute_matching_function(positions):
            return np.linalg.norm(positions - target_landmarks.flatten())**2

        def compute_gradient_matching_function(positions):
            return 2. * (positions - target_landmarks.flatten()) / sigma**2

        def set_opt(opt, energy, energy_regularizer, energy_l2, energy_l2_unscaled,
                    initial_momenta, registered_landmarks, time_evolution_momenta, time_evolution_positions):
            opt['energy'] = energy
            opt['energy_regularizer'] = energy_regularizer
            opt['energy_l2'] = energy_l2
            opt['energy_l2_unscaled'] = energy_l2_unscaled
            opt['initial_momenta'] = initial_momenta.reshape(input_landmarks.shape)
            opt['registered_landmarks'] = registered_landmarks.reshape(input_landmarks.shape)
            opt['time_evolution_momenta'] = time_evolution_momenta
            opt['time_evolution_positions'] = time_evolution_positions
            return opt

        momenta_time_dependent, positions_time_dependent = self.integrate_forward_Hamiltonian(
            initial_momenta, initial_positions)

        opt = set_opt({}, None, None, None, None, initial_momenta, initial_positions,
                      momenta_time_dependent, positions_time_dependent)

        reason_registration_ended = 'reached maximum number of iterations'

        start_time = time.perf_counter()

        res = {}

        def energy_func(initial_momenta, return_additional_infos=False, return_all_energies=False):
            momenta_time_dependent, positions_time_dependent = self.integrate_forward_Hamiltonian(
                initial_momenta, initial_positions)
            positions = positions_time_dependent[-1]

            energy_regularizer = self.compute_Hamiltonian(initial_momenta, positions)
            energy_l2_unscaled = compute_matching_function(positions)
            energy_l2 = 1. / sigma**2 * energy_l2_unscaled
            energy = energy_regularizer + energy_l2

            energies = {'energy': energy, 'energy_regularizer': energy_regularizer,
                        'energy_l2': energy_l2, 'energy_l2_unscaled': energy_l2_unscaled}
            if return_additional_infos:
                additional_infos = {'momenta_time_dependent': momenta_time_dependent,
                                    'positions_time_dependent': positions_time_dependent}
                if return_all_energies:
                    return additional_infos, energies
                return energy, additional_infos
            if return_all_energies:
                return energies
            return energy

        def gradient_func(momenta, momenta_time_dependent, positions_time_dependent):
            d_momenta_1 = self.integrate_forward_variational_Hamiltonian(momenta_time_dependent,
                                                                         positions_time_dependent)
            positions = positions_time_dependent[-1]
            grad_g = compute_gradient_matching_function(positions)
            grad = (self.K(initial_positions) @ momenta + d_momenta_1.T @ grad_g)

            return grad

        line_searcher = LineSearchAlgorithm(energy_func, gradient_func)
        optimizer = OptimizationAlgorithm(line_searcher)
        stepsize_controller = StepsizeControlAlgorithm(line_searcher)

        with self.logger.block("Perform image matching via geodesic shooting ..."):
            try:
                k = 0
                energy_did_not_decrease = 0
                x = res['x'] = initial_momenta
                energy, additional_infos = energy_func(res['x'], return_additional_infos=True)
                grad = gradient_func(x, **additional_infos)
                min_energy = energy

                while not (iterations is not None and k >= iterations):
                    x, energy, grad, current_stepsize = optimizer.step(x, energy, grad, parameters_line_search)

                    parameters_line_search = stepsize_controller.update(parameters_line_search, current_stepsize)

                    self.logger.info(f"iter: {k:3d}, energy: {energy:.4e}")

                    if min_energy >= energy:
                        res['x'] = x.copy()
                        min_energy = energy
                        if min_energy < energy_threshold:
                            self.logger.info(f"Energy below threshold of {energy_threshold}. "
                                             "Stopping ...")
                            reason_registration_ended = 'reached energy threshold'
                            break
                    else:
                        energy_did_not_decrease += 1

                    norm_gradient = np.linalg.norm(grad.flatten())
                    if norm_gradient < gradient_norm_threshold:
                        self.logger.warning(f"Gradient norm is {norm_gradient} "
                                            "and therefore below threshold. Stopping ...")
                        reason_registration_ended = 'reached gradient norm threshold'
                        break
                    if early_stopping is not None and energy_did_not_decrease >= early_stopping:
                        reason_registration_ended = 'early stopping due to non-decreasing energy'
                        break
                    k += 1
            except KeyboardInterrupt:
                self.logger.warning("Aborting registration ...")
                reason_registration_ended = 'manual abort'

        additional_infos, energies = energy_func(res['x'], return_additional_infos=True, return_all_energies=True)
        registered_landmarks = additional_infos['positions_time_dependent'][-1]
        set_opt(opt, energies['energy'], energies['energy_regularizer'], energies['energy_l2'],
                energies['energy_l2_unscaled'], res['x'], registered_landmarks,
                additional_infos['momenta_time_dependent'], additional_infos['positions_time_dependent'])

        elapsed_time = int(time.perf_counter() - start_time)

        self.logger.info(f"Finished registration ({reason_registration_ended}) ...")

        if opt['energy'] is not None:
            self.logger.info(f"Optimal energy: {opt['energy']:4.4f}")
            self.logger.info(f"Optimal l2-error (with scale): {opt['energy_l2']:4.4f}")
            self.logger.info(f"Optimal l2-error (without scale): {opt['energy_l2_unscaled']:4.4f}")
            self.logger.info(f"Optimal regularization: {opt['energy_regularizer']:4.4f}")

        opt['iterations'] = k
        opt['time'] = elapsed_time
        opt['reason_registration_ended'] = reason_registration_ended

        if return_all:
            return opt
        return res['x'].reshape((-1, self.dim))

    def compute_Hamiltonian(self, momenta, positions):
        """Computes the value of the Hamiltonian given positions and momenta.

        Parameters
        ----------
        momenta
            Array containing the (time-independent) momenta of the landmarks.
        positions
            Array containing the (time-independent) positions of the landmarks.

        Returns
        -------
        Value of the Hamiltonian.
        """
        assert momenta.shape == (self.size,)
        assert positions.shape == (self.size,)
        return 0.5 * momenta.T @ self.K(positions) @ momenta

    def integrate_forward_Hamiltonian(self, initial_momenta, initial_positions):
        """Performs forward integration of Hamiltonian equations to obtain time-dependent momenta
           and positions.

        Parameters
        ----------
        initial_momenta
            Array containing the initial momenta of the landmarks.
        initial_positions
            Array containing the initial positions of the landmarks.

        Returns
        -------
        Time-dependent momenta and positions.
        """
        assert initial_momenta.shape == (self.size,)
        assert initial_positions.shape == (self.size,)

        momenta = np.zeros((self.time_steps, self.size))
        positions = np.zeros((self.time_steps, self.size))
        momenta[0] = initial_momenta
        positions[0] = initial_positions

        for t in range(self.time_steps-1):
            momenta[t+1] = momenta[t] - self.dt * 0.5 * (momenta[t].T @ self.DK(positions[t]) @ momenta[t])
            positions[t+1] = positions[t] + self.dt * self.K(positions[t]) @ momenta[t]

        return momenta, positions

    def K(self, positions):
        """Computes matrix that contains (dim x dim)-blocks derived from the kernel.

        Parameters
        ----------
        positions
            Array containing the positions of the landmarks.

        Returns
        -------
        Matrix of shape (size x size).
        """
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
        """Computes derivative of the matrix K as a third order tensor.

        Parameters
        ----------
        positions
            Array containing the positions of the landmarks.

        Returns
        -------
        Tensor of shape (size x size x size).
        """
        mat = np.zeros((self.size, self.size, self.size))

        for i in range(self.size):
            modi = i % self.dim
            for j in range(self.size):
                modj = j % self.dim
                for k in range(self.size):
                    if i == k:
                        mat[i][j][k] += self.kernel.derivative_1(positions[i-modi:i+self.dim-modi],
                                                                 positions[j-modj:j+self.dim-modj],
                                                                 modi)
                    if j == k:
                        mat[i][j][k] += self.kernel.derivative_2(positions[i-modi:i+self.dim-modi],
                                                                 positions[j-modj:j+self.dim-modj],
                                                                 modj)

        return mat

    def integrate_forward_variational_Hamiltonian(self, momenta, positions):
        """Performs forward integration of Hamiltonian equations for derivatives of momenta
           and positions with respect to initial momenta and positions.

        Parameters
        ----------
        momenta
            Array containing the time-dependent momenta of the landmarks.
        positions
            Array containing the time-dependent positions of the landmarks.

        Returns
        -------
        Derivative of the momenta at final time t=1.
        """
        assert positions.shape == (self.time_steps, self.size)
        assert momenta.shape == (self.time_steps, self.size)

        d_positions = np.zeros((self.time_steps, self.size, self.size))
        d_momenta = np.zeros((self.time_steps, self.size, self.size))
        d_momenta[0] = np.eye(self.size)

        for t in range(self.time_steps-1):
            d_positions[t+1] = d_positions[t] + self.dt * (self.DK(positions[t]) @ d_positions[t] @ momenta[t]
                                                           + self.K(positions[t]) @ d_momenta[t])
            d_momenta[t+1] = d_momenta[t] - self.dt * (d_momenta[t] @ self.DK(positions[t]) @ positions[t]
                                                       + positions[t] @ self.DK(positions[t]) @ d_momenta[t])
#                + positions[t] @ @ positions[t]  # maybe a term is missing here...

        return d_momenta[-1]

    def get_vector_fields(self, momenta, positions, grid):
        """Evaluates vector field given by positions and momenta at grid points.

        Parameters
        ----------
        momenta
            Array containing the time-dependent momenta of the landmarks.
        positions
            Array containing the time-dependent positions of the landmarks.
        grid
            Array containing the grid points.

        Returns
        -------
        Vector field at the grid points.
        """
        assert momenta.shape == (self.time_steps, self.size)
        assert positions.shape == (self.time_steps, self.size)
        assert grid.ndim == 2
        assert grid.shape[1] == self.dim

        vector_fields = np.zeros((self.time_steps, *grid.shape))

        for t in range(self.time_steps):
            vf_func = construct_vector_field(momenta[t].reshape((-1, self.dim)),
                                             positions[t].reshape((-1, self.dim)),
                                             kernel=self.kernel)
            for i, pos in enumerate(grid):
                vector_fields[t][i] = vf_func(pos)

        return vector_fields

    def compute_time_evolution_of_diffeomorphisms(self, initial_momenta, initial_positions, grid):
        """Performs forward integration of diffeomorphism on given grid using the given
           initial momenta and positions.

        Parameters
        ----------
        initial_momenta
            Array containing the initial momenta of the landmarks.
        initial_positions
            Array containing the initial positions of the landmarks.
        grid
            Array containing the grid points.

        Returns
        -------
        Array containing the diffeomorphism at the different time instances.
        """
        assert initial_momenta.shape == initial_positions.shape
        assert grid.ndim == 2
        assert grid.shape[1] == self.dim

        momenta, positions = self.integrate_forward_Hamiltonian(initial_momenta.flatten(), initial_positions.flatten())
        vector_fields = self.get_vector_fields(momenta, positions, grid)
        diffeomorphisms = np.zeros((self.time_steps, *grid.shape))
        diffeomorphisms[0] = grid

        assert diffeomorphisms.shape == (self.time_steps, *grid.shape)

        for t in range(self.time_steps-1):
            # composition with diffeomorphisms[t]!!!
            diffeomorphisms[t+1] = (diffeomorphisms[t] + self.dt * vector_fields[t].reshape(diffeomorphisms[t].shape))

        return diffeomorphisms

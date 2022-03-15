import time
import numpy as np

from copy import deepcopy

from geodesic_shooting.utils import sampler, grid
from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer
from geodesic_shooting.utils.optim import GradientDescentOptimizer, ArmijoLineSearch, PatientStepsizeController
from geodesic_shooting.core import ScalarFunction, VectorField, TimeDependentVectorField


class GeodesicShooting:
    """Class that implements large deformation metric mappings via geodesic shooting.

    Based on:
    Geodesic Shooting for Computational Anatomy.
    Miller, Trouvé, Younes, 2006
    """
    def __init__(self, alpha=6., exponent=2., dim=2, shape=(100, 100), log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        alpha
            Parameter for biharmonic regularizer.
        exponent
            Parameter for biharmonic regularizer.
        dim
            Dimension of the input and target images (set automatically when calling `register`).
        shape
            Shape of the input and target images (set automatically when calling `register`).
        log_level
            Verbosity of the logger.
        """
        self.regularizer = BiharmonicRegularizer(alpha, exponent)

        self.time_steps = 30
        self.shape = shape
        self.dim = dim
        assert self.dim == len(self.shape)

        self.logger = getLogger('geodesic_shooting', level=log_level)

    def register(self, input_, target, time_steps=30, sigma=1.,
                 OptimizationAlgorithm=GradientDescentOptimizer, iterations=1000, early_stopping=10,
                 initial_vector_field=None, LineSearchAlgorithm=ArmijoLineSearch,
                 parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1.,
                                         'max_num_search_steps': 10},
                 StepsizeControlAlgorithm=PatientStepsizeController,
                 energy_threshold=1e-6, gradient_norm_threshold=1e-6,
                 return_all=False, log_summary=True):
        """Performs actual registration according to LDDMM algorithm with time-varying vector
           fields that are chosen via geodesics.

        Parameters
        ----------
        input_
            Input image as array.
        target
            Target image as array.
        time_steps
            Number of discrete time steps to perform.
        sigma
            Weight for the similarity measurement (L2 difference of the target and the registered
            image); the smaller sigma, the larger the influence of the L2 loss.
        OptimizationAlgorithm
            Algorithm to use for optimization during registration. Should be a class and not an
            instance. The class should derive from `BaseOptimizer`.
        iterations
            Number of iterations of the optimizer to perform. The value `None` is also possible
            to not bound the number of iterations.
        early_stopping
            Number of iterations with non-decreasing energy after which to stop registration.
            If `None`, no early stopping is used.
        initial_vector_field
            Used as initial guess for the initial vector field (will be 0 if None is passed).
        LineSearchAlgorithm
            Algorithm to use as line search method during optimization. Should be a class and not
            an instance. The class should derive from `BaseLineSearch`.
        parameters_line_search
            Additional parameters for the line search algorithm
            (e.g. minimal and maximal stepsize, ...).
        StepsizeControlAlgorithm
            Algorithm to use for adjusting the minimal and maximal stepsize (or other parameters
            of the line search algorithm that are prescribed in `parameters_line_search`).
            The class should derive from `BaseStepsizeController`.
        energy_threshold
            If the energy drops below this threshold, the registration is stopped.
        gradient_norm_threshold
            If the norm of the gradient drops below this threshold, the registration is stopped.
        return_all
            Determines whether or not to return all information or only the initial vector field
            that led to the best registration result.
        log_summary
            Determines whether or not to print a summary of the registration results to the
            console.

        Returns
        -------
        Either the best initial vector field (if return_all is False) or a dictionary consisting
        of the registered image, the velocities, the energies, the flows and inverse flows, the
        forward-pushed input and the back-pulled target at all time instances (if return_all is
        True).
        """
        assert isinstance(time_steps, int) and time_steps > 0
        assert iterations is None or (isinstance(iterations, int) and iterations > 0)
        assert sigma > 0
        assert (isinstance(early_stopping, int) and early_stopping > 0) or early_stopping is None

        assert isinstance(input_, ScalarFunction)
        assert isinstance(target, ScalarFunction)
        assert input_.full_shape == target.full_shape

        # function to compute the L2-error between a given image and the target
        def compute_energy(image):
            return np.sum(((image - target)**2).to_numpy())

        # function to compute the gradient of the overall energy function
        # with respect to the final vector field
        def compute_grad_energy(image):
            return self.regularizer.cauchy_navier_squared_inverse(image.grad * (image - target)[..., np.newaxis])

        # set up variables
        self.time_steps = time_steps
        self.shape = input_.spatial_shape
        self.dim = input_.dim

        # define initial vector fields
        if initial_vector_field is None:
            initial_vector_field = VectorField(self.shape)
        else:
            if not isinstance(initial_vector_field, VectorField):
                initial_vector_field = VectorField(data=initial_vector_field)
        assert isinstance(initial_vector_field, VectorField)
        assert initial_vector_field.full_shape == (*self.shape, self.dim)

        # updates dictionary with (current) optimal values
        def set_opt(opt, energy, energy_regularizer, energy_intensity, energy_intensity_unscaled,
                    transformed_input, initial_vector_field, flow, vector_fields):
            opt['energy'] = energy
            opt['energy_regularizer'] = energy_regularizer
            opt['energy_intensity'] = energy_intensity
            opt['energy_intensity_unscaled'] = energy_intensity_unscaled
            opt['transformed_input'] = transformed_input
            opt['initial_vector_field'] = initial_vector_field
            opt['flow'] = flow
            opt['vector_fields'] = vector_fields
            return opt

        vector_fields = self.integrate_forward_vector_field(initial_vector_field)
        opt = set_opt({}, None, None, None, None, input_, initial_vector_field,
                      self.integrate_forward_flow(vector_fields), vector_fields)

        reason_registration_ended = 'reached maximum number of iterations'

        start_time = time.perf_counter()

        res = {}

        # function that computes the energy
        def energy_func(v0, return_additional_infos=False, return_all_energies=False):
            # integrate initial vector field forward in time
            vector_fields = self.integrate_forward_vector_field(v0)

            # compute forward flows according to the vector fields
            flow = self.integrate_forward_flow(vector_fields)

            # push-forward input_ image
            forward_pushed_input = self.push_forward(input_, flow)

            # compute the current energy consisting of intensity difference
            # and regularization
            energy_regularizer = self.regularizer.cauchy_navier(v0).norm
            energy_intensity_unscaled = compute_energy(forward_pushed_input)
            energy_intensity = 1 / sigma**2 * energy_intensity_unscaled
            energy = energy_regularizer + energy_intensity

            if return_additional_infos:
                return energy, {'vector_fields': vector_fields, 'forward_pushed_input': forward_pushed_input}
            if return_all_energies:
                return {'energy': energy, 'energy_regularizer': energy_regularizer,
                        'energy_intensity': energy_intensity, 'energy_intensity_unscaled': energy_intensity_unscaled}
            return energy

        # function that computes the gradient of the energy
        def gradient_func(v0, vector_fields, forward_pushed_input):
            # compute gradient of the intensity difference
            gradient_l2_energy = compute_grad_energy(forward_pushed_input) / sigma**2

            # compute gradient of the intensity difference with respect to the initial vector field
            gradient_initial_vector = -self.integrate_backward_adjoint_Jacobi_field(gradient_l2_energy, vector_fields)

            return gradient_initial_vector

        line_searcher = LineSearchAlgorithm(energy_func, gradient_func)
        optimizer = OptimizationAlgorithm(line_searcher)
        stepsize_controller = StepsizeControlAlgorithm(line_searcher)

        # beginning of the main registration routine
        with self.logger.block("Perform image matching via geodesic shooting ..."):
            try:
                # set initial values
                k = 0
                energy_did_not_decrease = 0
                x = res['x'] = initial_vector_field
                energy, additional_infos = energy_func(res['x'], return_additional_infos=True)
                grad = gradient_func(res['x'], **additional_infos)
                min_energy = energy

                # registration iteration
                while not (iterations is not None and k >= iterations):
                    # perform optimization step
                    x, energy, grad, current_stepsize = optimizer.step(x, energy, grad, parameters_line_search)
                    # update the stepsize controller
                    parameters_line_search = stepsize_controller.update(parameters_line_search, current_stepsize)

                    self.logger.info(f"iter: {k:3d}, energy: {energy:.4e}")

                    # check if objective function value decreased
                    if min_energy >= energy:
                        res['x'] = deepcopy(x)
                        min_energy = energy
                        if min_energy < energy_threshold:
                            self.logger.info(f"Energy below threshold of {energy_threshold}. "
                                             "Stopping ...")
                            reason_registration_ended = 'reached energy threshold'
                            break
                    else:
                        energy_did_not_decrease += 1

                    # check the norm of the gradient
                    norm_gradient = grad.norm
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

            # compute time-dependent vector field from optimal initial vector field
            vector_fields = self.integrate_forward_vector_field(res['x'])

            # compute forward flows according to the vector fields
            flow = self.integrate_forward_flow(vector_fields)

            # push-forward input-image
            transformed_input = self.push_forward(input_, flow)

        # update optimal values
        energies = energy_func(res['x'], return_all_energies=True)
        set_opt(opt, energies['energy'], energies['energy_regularizer'], energies['energy_intensity'],
                energies['energy_intensity_unscaled'], transformed_input, res['x'], flow, vector_fields)

        elapsed_time = int(time.perf_counter() - start_time)

        self.logger.info(f"Finished registration ({reason_registration_ended}) ...")

        if opt['energy'] is not None:
            self.logger.info(f"Optimal energy: {opt['energy']:4.4f}")

        if opt['initial_vector_field'] is not None:
            # compute the length of the path on the manifold;
            # this step only requires the initial vector due to conservation of momentum
            length = self.regularizer.cauchy_navier(opt['initial_vector_field']).norm
        else:
            length = 0.0

        opt['input'] = input_
        opt['target'] = target

        opt['length'] = length
        opt['iterations'] = k
        opt['time'] = elapsed_time
        opt['reason_registration_ended'] = reason_registration_ended

        if log_summary:
            self.summarize_results(opt)

        if return_all:
            return opt
        return initial_vector_field

    def summarize_results(self, results):
        """Log a summary of the results to the console.

        Parameters
        ----------
        results
            Dictionary with the results obtained from the `register`-function.
        """
        self.logger.info("")
        self.logger.info("Registration summary")
        self.logger.info("====================")
        self.logger.info(f"Registration finished after {results['iterations']} iterations.")
        self.logger.info(f"Registration took {results['time']} seconds.")
        self.logger.info(f"Reason for the registration algorithm to stop: {results['reason_registration_ended']}.")
        self.logger.info("Relative norm of difference: "
                         f"{(results['target'] - results['transformed_input']).norm / results['target'].norm}")

    def integrate_forward_flow(self, vector_fields):
        """Computes forward integration according to given vector fields.

        Parameters
        ----------
        vector_fields
            Sequence of vector fields (i.e. time-depending vector field).

        Returns
        -------
        Array containing the flow at the final time.
        """
        # make identity grid
        identity_grid = grid.coordinate_grid(self.shape)

        # initial flow is the identity mapping
        flow = identity_grid.copy()

        # perform forward integration
        for t in range(0, self.time_steps-1):
            flow = sampler.sample(flow, identity_grid - vector_fields[t])

        return flow

    def push_forward(self, image, flow):
        """Pushes forward an image along a flow.

        Parameters
        ----------
        image
            `ScalarFunction` to push forward.
        flow
            `VectorField` containing the flow according to which to push the input forward.

        Returns
        -------
        Array with the forward-pushed image.
        """
        return sampler.sample(image, flow)

    def integrate_forward_vector_field(self, initial_vector_field):
        """Performs forward integration of the initial vector field.

        Hint: See "Finite-Dimensional Lie Algebras for Fast Diffeomorphic Image Registration"
        by Miaomiao Zhang and P. Thomas Fletcher, Section 2, Equation (3), or "Data-driven
        Model Order Reduction For Diffeomorphic Image Registration" by Jian Wang, Wei Xing,
        Robert M. Kirby, and Miaomiao Zhang, Section 2, Equation (3), for more information
        on the equations used here.

        Parameters
        ----------
        initial_vector_field
            Initial `VectorField` to integrate forward.

        Returns
        -------
        Sequence of vector fields obtained via forward integration of the initial vector field.
        """
        # set up time-dependent vector field and set initial value
        vector_fields = TimeDependentVectorField(self.shape, self.time_steps)
        vector_fields[0] = initial_vector_field

        # einsum strings used for multiplication of (transposed) Jacobian matrix of vector fields
        einsum_string = '...lk,...k->...l'
        einsum_string_transpose = '...kl,...k->...l'

        # perform forward in time integration of initial vector field
        for t in range(0, self.time_steps-1):
            # compute the current momentum
            momentum_t = self.regularizer.cauchy_navier(vector_fields[t])
            # compute the gradient (Jacobian) of the current momentum
            grad_mt = momentum_t.grad
            # compute the gradient (Jacobian) of the current vector field
            grad_vt = vector_fields[t].grad
            # compute the divergence of the current vector field
            div_vt = np.sum(np.array([grad_vt[..., d, d] for d in range(self.dim)]), axis=0)
            # compute the right hand side, i.e. Dv^T m + Dm v + m div v
            rhs = (np.einsum(einsum_string_transpose, grad_vt, momentum_t.to_numpy())
                   + np.einsum(einsum_string, grad_mt, vector_fields[t].to_numpy())
                   + momentum_t.to_numpy() * div_vt[..., np.newaxis])
            rhs = VectorField(data=rhs)
            # perform the explicit Euler integration step
            vector_fields[t+1] = vector_fields[t] - self.regularizer.cauchy_navier_squared_inverse(rhs)/self.time_steps

        return vector_fields

    def integrate_backward_adjoint_Jacobi_field(self, gradient_l2_energy, vector_fields):
        """Performs backward integration of the adjoint jacobi field equations.

        Hint: See "Finite-Dimensional Lie Algebras for Fast Diffeomorphic Image Registration"
        by Miaomiao Zhang and P. Thomas Fletcher, Section 4.2, for more information on the
        equations used here.

        Parameters
        ----------
        gradient_l2_energy
            Array containing the gradient of the L2 energy functional.
        vector_fields
            Sequence of vector fields (i.e. time-dependent vector field) to integrate backwards.

        Returns
        -------
        Gradient of the energy with respect to the initial vector field.
        """
        # introduce adjoint variables
        v_old = gradient_l2_energy
        delta_v_old = VectorField(v_old.spatial_shape)
        delta_v = delta_v_old.copy()

        # einsum strings used for multiplication of (transposed) Jacobian matrix of vector fields
        einsum_string = '...lk,...l->...k'
        einsum_string_transpose = '...kl,...l->...k'

        # perform backward in time integration of the gradient of the energy function
        for t in range(self.time_steps-2, -1, -1):
            # get gradient of the current vector field
            grad_vector_fields = vector_fields[t].grad
            # get divergence of the current vector field
            div_vector_fields = np.sum(np.array([grad_vector_fields[..., d, d]
                                                 for d in range(self.dim)]), axis=0)
            # get momentum corresponding to the adjoint variable `v_old`
            regularized_v = self.regularizer.cauchy_navier(v_old)
            # get gradient of the momentum of `v_old`
            grad_regularized_v = regularized_v.grad

            # update adjoint variable `v_old`
            rhs_v = - self.regularizer.cauchy_navier_squared_inverse(
                VectorField(data=np.einsum(einsum_string_transpose, grad_vector_fields, regularized_v.to_numpy()))
                + VectorField(data=np.einsum(einsum_string, grad_regularized_v, vector_fields[t].to_numpy()))
                + regularized_v * div_vector_fields[..., np.newaxis])
            v_old = v_old - rhs_v / self.time_steps

            # get gradient of the adjoint variable `delta_v`
            grad_delta_v = delta_v.grad
            # get divergence of the adjoint variable `delta_v`
            div_delta_v = np.sum(np.array([grad_delta_v[..., d, d]
                                           for d in range(self.dim)]), axis=0)
            # get momentum corresponding to the current vector field
            regularized_vector_fields = self.regularizer.cauchy_navier(vector_fields[t])
            # get gradient of the momentum of the current vector field
            grad_regularized_vector_fields = regularized_vector_fields.grad
            # update the adjoint variable `delta_v`
            rhs_delta_v = (- v_old
                           - (np.einsum(einsum_string, grad_vector_fields, delta_v.to_numpy())
                              - np.einsum(einsum_string, grad_delta_v, vector_fields[t].to_numpy()))
                           + self.regularizer.cauchy_navier_squared_inverse(
                               VectorField(data=np.einsum(einsum_string_transpose,
                                                          grad_delta_v,
                                                          regularized_vector_fields.to_numpy()))
                               + VectorField(data=np.einsum(einsum_string,
                                                            grad_regularized_vector_fields,
                                                            delta_v.to_numpy()))
                               + regularized_vector_fields * div_delta_v[..., np.newaxis]))
            delta_v = delta_v_old - rhs_delta_v / self.time_steps
            delta_v_old = delta_v

        # return adjoint variable `delta_v` that corresponds to the gradient
        # of the objective function at the initial time instance
        return delta_v

# -*- coding: utf-8 -*-
import time
import numpy as np

from copy import deepcopy

import scipy.optimize as optimize

from geodesic_shooting.core import ScalarFunction, VectorField, TimeDependentVectorField
from geodesic_shooting.utils import sampler, grid
from geodesic_shooting.utils.grad import finite_difference_matrix, gradient_matrix, divergence_matrix
from geodesic_shooting.utils.helper_functions import tuple_product
from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer
from geodesic_shooting.utils.time_integration import RK4


class ReducedGeodesicShooting:
    """Class that implements large deformation metric mappings via geodesic shooting.

    Based on:
    Geodesic Shooting for Computational Anatomy.
    Miller, Trouvé, Younes, 2006
    """
    def __init__(self, rb_vector_fields=None, alpha=6., exponent=2., time_integrator=RK4, time_steps=30,
                 sampler_options={'order': 1, 'mode': 'edge'}, precomputed_quantities={}, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        rb_vector_fields
            Reduced basis for initial vector fields (for instance computed via proper orthogonal
            decomposition of high-dimensional vector fields).
        alpha
            Parameter for biharmonic regularizer.
        exponent
            Parameter for biharmonic regularizer.
        time_integrator
            Method to use for time integration.
        time_steps
            Number of time steps performed during forward and backward integration.
        sampler_options
            Additional options to pass to the sampler.
        log_level
            Verbosity of the logger.
        """
        assert rb_vector_fields is not None or 'rb_vector_fields' in precomputed_quantities
        if rb_vector_fields is not None:
            self.rb_vector_fields = rb_vector_fields
        else:
            self.rb_vector_fields = precomputed_quantities['rb_vector_fields']
        assert isinstance(self.rb_vector_fields, list)
        assert all([isinstance(v, VectorField) for v in self.rb_vector_fields])
        self.rb_size = len(self.rb_vector_fields)
        assert self.rb_size > 0

        self.regularizer = BiharmonicRegularizer(alpha, exponent)

        self.time_integrator = time_integrator
        self.time_steps = time_steps
        self.dt = 1. / self.time_steps

        self.size = tuple_product(self.rb_vector_fields[0].full_shape)
        self.shape = self.rb_vector_fields[0].spatial_shape
        self.dim = self.rb_vector_fields[0].dim

        self.sampler_options = sampler_options

        self.logger = getLogger('reduced_geodesic_shooting', level=log_level)

        self.logger.info("Initialize matrices of regularizer ...")
        self.regularizer.init_matrices(self.shape)

        matrices_labels = ['matrices_forward', 'matrices_backward_1',
                           'matrices_backward_2', 'matrices_backward_3']

        if any([label not in precomputed_quantities for label in matrices_labels]):
            D = finite_difference_matrix(self.shape)
            assert D.shape == (self.dim * self.size, self.dim * self.size)
            div = divergence_matrix(self.shape)
            assert div.shape == (self.dim * self.size, self.dim * self.size)
            L = self.regularizer.cauchy_navier_matrix
            assert L.shape == (self.dim * self.size, self.dim * self.size)
            K = self.regularizer.cauchy_navier_inverse_matrix
            assert K.shape == (self.dim * self.size, self.dim * self.size)

            self.matrices_forward = []
            self.matrices_backward_1 = []
            self.matrices_backward_2 = []
            self.matrices_backward_3 = []

            U = np.array([v.to_numpy().squeeze() for v in self.rb_vector_fields]).T

            UTK = U.T.dot(K)
            DL = D.dot(L)
            DTU = D.T.dot(U)
            DLU = DL.dot(U)
            divU = div.dot(U)
            DU = D.dot(U)

            self.logger.info("Compute reduced matrices ...")

            for j in range(self.rb_size):
                matrix_forward = np.zeros((self.rb_size, self.rb_size))
                matrix_backward_1 = np.zeros((self.rb_size, self.rb_size))
                matrix_backward_2 = np.zeros((self.rb_size, self.rb_size))
                matrix_backward_3 = np.zeros((self.rb_size, self.rb_size))

                for i in range(self.dim * self.size):
                    unit_vector = np.zeros(self.dim * self.size)
                    unit_vector[i] = 1.
                    matrix_forward += - U[i, j] * (UTK.dot(np.diag(L[:, i])).dot(DTU)
                                                   + UTK.dot(np.diag(unit_vector)).dot(DLU)
                                                   + UTK.dot(np.diag(L[:, i])).dot(divU))
                    matrix_backward_1 += U[i, j] * (UTK.dot(np.diag(L[:, i])).dot(DTU)
                                                    + UTK.dot(np.diag(L[:, i])).dot(divU))
                    matrix_backward_2 += UTK.dot(np.diag(unit_vector)).dot(DLU) * U[i, j]
                    matrix_backward_3 += U.T.dot(np.diag(unit_vector)).dot(DU) * U[i, j]

                self.matrices_forward.append(matrix_forward)
                self.matrices_backward_1.append(matrix_backward_1)
                self.matrices_backward_2.append(matrix_backward_2)
                self.matrices_backward_3.append(matrix_backward_3)
        else:
            assert all([label in precomputed_quantities for label in matrices_labels])
            self.logger.info("Use precomputed reduced matrices ...")
            for label in matrices_labels:
                setattr(self, label, precomputed_quantities[label])

        self.logger.info("Finished setting up everything ...")

    def get_reduced_quantities(self):
        return {'rb_vector_fields': self.rb_vector_fields,
                'matrices_forward': self.matrices_forward,
                'matrices_backward_1': self.matrices_backward_1,
                'matrices_backward_2': self.matrices_backward_2,
                'matrices_backward_3': self.matrices_backward_3}

    def __str__(self):
        return (f"Alpha: {self.regularizer.alpha}\nExponent: {self.regularizer.exponent}\n"
                f"Time integrator: {self.time_integrator.__name__}\nTime steps: {self.time_steps}")

    def register(self, input_, target, sigma=1.,
                 optimization_method='L-BFGS-B',
                 optimizer_options={'disp': True},
                 initial_vector_field=None,
                 return_all=False, log_summary=True):
        """Performs actual registration according to LDDMM algorithm with time-varying vector
           fields that are chosen via geodesics.

        Parameters
        ----------
        input_
            Input image as array.
        target
            Target image as array.
        sigma
            Weight for the similarity measurement (L2 difference of the target and the registered
            image); the smaller sigma, the larger the influence of the L2 loss.
        optimization_method
            Optimizer from `scipy`, see `method` under
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        optimizer_options
            Additional options passed to the `scipy.optimize.minimize`-function, see `options` under
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        initial_vector_field
            Used as initial guess for the initial vector field (will be 0 if None is passed).
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
        assert sigma > 0

        assert isinstance(input_, ScalarFunction)
        assert isinstance(target, ScalarFunction)
        assert input_.full_shape == target.full_shape

        def compute_energy(image):
            return np.sum((image - target).to_numpy()**2)

        def compute_grad_energy(image):
            """ Not 100% sure whether this is correct... """
            return self.regularizer.cauchy_navier_inverse_matrix.dot(
                (image.grad.to_numpy().reshape((self.dim, self.size))
                 * (image - target)[np.newaxis, ...]).reshape(self.dim * self.size))

        # set up variables
        assert self.shape == input_.spatial_shape
        assert self.dim == input_.dim

        # define initial vector fields
        if initial_vector_field is None:
            initial_vector_field = np.zeros(self.rb_size)
        else:
            if not isinstance(initial_vector_field, VectorField):
                initial_vector_field = VectorField(data=initial_vector_field)

        opt = {'input': input_, 'target': target}

        reason_registration_ended = 'reached maximum number of iterations'

        start_time = time.perf_counter()

        # function that computes the energy
        def energy_and_gradient(v0):
            # integrate initial vector field forward in time
            vector_fields = self.integrate_forward_vector_field(v0)

            # compute forward flows according to the vector fields
            flow = self.integrate_forward_flow(vector_fields)

            # push-forward input_ image
            forward_pushed_input = input_.push_forward(flow)

            # compute the current energy consisting of intensity difference
            # and regularization
            vf = VectorField(self.shape)
            for v, r in zip(self.rb_vector_fields, v0):
                vf += r * v
            energy_regularizer = self.regularizer.cauchy_navier(vf).norm
            energy_intensity_unscaled = compute_energy(forward_pushed_input)
            energy_intensity = 1 / sigma**2 * energy_intensity_unscaled
            energy = energy_regularizer + energy_intensity

            # compute gradient of the intensity difference
            gradient_l2_energy = compute_grad_energy(forward_pushed_input) / sigma**2

            # compute gradient of the intensity difference with respect to the initial vector field
            gradient_initial_vector = -self.integrate_backward_adjoint(gradient_l2_energy, vector_fields)

            return energy, gradient_initial_vector

        def save_current_state(x):
            opt['x'] = x

        # use scipy optimizer for minimizing energy function
        with self.logger.block("Perform image matching via geodesic shooting ..."):
            res = optimize.minimize(energy_and_gradient, initial_vector_field,
                                    method=optimization_method, jac=True, options=optimizer_options,
                                    callback=save_current_state)

        # compute time-dependent vector field from optimal initial vector field
        vector_fields = self.integrate_forward_vector_field(res['x'])

        # compute forward flows according to the vector fields
        flow = self.integrate_forward_flow(vector_fields)

        # push-forward input-image
        transformed_input = input_.push_forward(flow)

        full_initial_vector_field = VectorField(self.shape)
        for v, r in zip(self.rb_vector_fields, res['x']):
            full_initial_vector_field += r * v
        opt['initial_vector_field'] = full_initial_vector_field
        opt['transformed_input'] = transformed_input
        opt['flow'] = flow
        opt['vector_fields'] = vector_fields

        elapsed_time = int(time.perf_counter() - start_time)

        self.logger.info(f"Finished registration ({reason_registration_ended}) ...")

        if opt['initial_vector_field'] is not None:
            # compute the length of the path on the manifold;
            # this step only requires the initial vector due to conservation of momentum
            length = self.regularizer.cauchy_navier(opt['initial_vector_field']).norm
        else:
            length = 0.0

        opt['length'] = length
        opt['iterations'] = res['nit']
        opt['time'] = elapsed_time
        opt['reason_registration_ended'] = res['message']

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
            vf = VectorField(self.shape)
            for v, r in zip(self.rb_vector_fields, vector_fields[t]):
                vf += r * v
            flow = sampler.sample(flow, identity_grid - vf, sampler_options=self.sampler_options)

        return flow

    def image_grad(self, image):
        """Computes the gradients of the given image.
        Parameters
        ----------
        image
            Array containing the (forward) pushed image.
        Returns
        -------
        Array with the gradients of the input image.
        """
        assert image.shape == (self.size, )
        gradient_mat = gradient_matrix(self.shape)
        assert gradient_mat.shape == (self.dim * self.size, self.size)
        return gradient_mat.dot(image)

    def integrate_forward_vector_field(self, initial_vector_field):
        """Performs forward integration of the initial vector field.

        Parameters
        ----------
        initial_vector_field
            Array with the initial vector field to integrate forward.

        Returns
        -------
        Sequence of vector fields obtained via forward integration of the initial vector field.
        """
        vector_fields = np.zeros((self.time_steps, self.rb_size), dtype=np.double)
        vector_fields[0] = initial_vector_field

        for t in range(0, self.time_steps-2):
            v = vector_fields[t]
            assert v.shape == (self.rb_size, )
            rhs = np.sum(np.array([mat.dot(v) * v_i
                                   for mat, v_i in zip(self.matrices_forward, v)]),
                         axis=0)
            assert rhs.shape == (self.rb_size, )
            vector_fields[t+1] = vector_fields[t] + rhs / self.time_steps
            assert vector_fields[t+1].shape == (self.rb_size, )

        return vector_fields

    def integrate_backward_adjoint(self, gradient_l2_energy, vector_fields):
        """Performs backward integration of the adjoint Jacobi field equations.
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
        v_adjoint = np.array([v.to_numpy().flatten().dot(gradient_l2_energy) for v in self.rb_vector_fields])
        assert v_adjoint.shape == (self.rb_size, )
        delta_v = np.zeros(v_adjoint.shape, dtype=np.double)
        assert delta_v.shape == (self.rb_size, )

        for t in range(self.time_steps-2, -1, -1):
            v = vector_fields[t]
            assert v.shape == (self.rb_size, )
            rhs_v = - (np.sum(np.array([mat.dot(v) * delta_v_i for mat, delta_v_i in
                                        zip(self.matrices_backward_1, delta_v)]),
                              axis=0) +
                       np.sum(np.array([mat.dot(delta_v) * v_i for mat, v_i in
                                        zip(self.matrices_backward_2, v)]),
                              axis=0))
            assert rhs_v.shape == (self.rb_size, )
            v_adjoint = v_adjoint - rhs_v / self.time_steps
            assert v_adjoint.shape == (self.rb_size, )

            rhs_delta_v = (- v_adjoint
                           - (np.sum(np.array([mat.dot(v) * delta_v_i for mat, delta_v_i in
                                               zip(self.matrices_backward_3, delta_v)]),
                                     axis=0) -
                              np.sum(np.array([mat.dot(delta_v) * v_i for mat, v_i in
                                               zip(self.matrices_backward_3, v)]),
                                     axis=0))
                           + (np.sum(np.array([mat.dot(delta_v) * v_i for mat, v_i in
                                               zip(self.matrices_backward_1, v)]),
                                     axis=0) +
                              np.sum(np.array([mat.dot(v) * delta_v_i for mat, delta_v_i in
                                               zip(self.matrices_backward_2, delta_v)]),
                                     axis=0)))
            assert rhs_delta_v.shape == (self.rb_size, )
            delta_v = delta_v - rhs_delta_v / self.time_steps
            assert delta_v.shape == (self.rb_size, )

        return delta_v

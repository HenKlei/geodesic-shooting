# -*- coding: utf-8 -*-
import time
import numpy as np
import scipy.optimize as optimize

from geodesic_shooting.core import ScalarFunction, VectorField, TimeDependentVectorField
from geodesic_shooting.utils.logger import getLogger
from geodesic_shooting.utils.regularizer import BiharmonicRegularizer
from geodesic_shooting.utils.time_integration import RK4


class GeodesicShooting:
    """Class that implements large deformation metric mappings via geodesic shooting.

    Based on:
    Geodesic Shooting for Computational Anatomy.
    Miller, Trouvé, Younes, 2006
    """
    def __init__(self, alpha=0.1, exponent=1, gamma=1., time_integrator=RK4, time_steps=30,
                 fourier=False, spatial_shape=None, sampler_options={'order': 1, 'mode': 'edge'},
                 log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        alpha
            Parameter for biharmonic regularizer.
        exponent
            Parameter for biharmonic regularizer.
        gamma
            Parameter for biharmonic regularizer.
        time_integrator
            Method to use for time integration.
        time_steps
            Number of time steps performed during forward and backward integration.
        fourier
            Parameter for biharmonic regularizer.
        spatial_shape
            Parameter for biharmonic regularizer.
        sampler_options
            Additional options to pass to the sampler.
        log_level
            Verbosity of the logger.
        """
        self.regularizer = BiharmonicRegularizer(alpha, exponent, gamma, fourier=fourier, spatial_shape=spatial_shape)

        self.time_integrator = time_integrator
        self.time_steps = time_steps
        self.dt = 1. / self.time_steps

        self.sampler_options = sampler_options

        self.logger = getLogger('geodesic_shooting', level=log_level)

    def __str__(self):
        return (f"{self.__class__.__name__}:\n"
                f"\tRegularizer:\t{self.regularizer}\n"
                f"\tTime integrator: {self.time_integrator.__name__}\n"
                f"\tTime steps: {self.time_steps}\n"
                f"\tSampler options: {self.sampler_options}")

    def register(self, template, target, sigma=0.1, optimization_method='L-BFGS-B', optimizer_options={'disp': True},
                 initial_vector_field=None, restriction=np.s_[...], return_all=False, log_summary=True):
        """Performs actual registration according to LDDMM algorithm with time-varying vector
           fields that are chosen via geodesics.

        Parameters
        ----------
        template
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
        restriction
            Slice that can be used to restrict the domain on which to compute the error
            and its gradient.
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

        assert isinstance(template, ScalarFunction)
        assert isinstance(target, ScalarFunction)
        assert template.full_shape == target.full_shape

        self.regularizer.init_matrices(template.spatial_shape)

        inverse_mask = np.ones(template.spatial_shape, bool)
        inverse_mask[restriction] = 0

        # function to compute the L2-error between a given image and the target
        def compute_energy(image):
            return (image-target).get_norm(restriction=restriction)**2

        # function to compute the gradient of the overall energy function
        # with respect to the final vector field
        def compute_grad_energy(image):
            grad_diff = image.grad * (image - target)[..., np.newaxis]
            grad_diff[inverse_mask] = 0.
            return VectorField(data=2. * self.regularizer.cauchy_navier_inverse(grad_diff))

        # set up variables
        self.shape = template.spatial_shape
        self.dim = template.dim

        # define initial vector fields
        if initial_vector_field is None:
            initial_vector_field = VectorField(self.shape)
        else:
            if not isinstance(initial_vector_field, VectorField):
                initial_vector_field = VectorField(data=initial_vector_field)
        assert isinstance(initial_vector_field, VectorField)
        assert initial_vector_field.full_shape == (*self.shape, self.dim)

        opt = {'input': template, 'target': target}

        start_time = time.perf_counter()

        # function that computes the energy
        def energy_and_gradient(v0, compute_grad=True, return_all_energies=False):
            v0 = VectorField(data=v0.reshape((*self.shape, self.dim)))
            # integrate initial vector field forward in time
            vector_fields = self.integrate_forward_vector_field(v0)

            # compute forward flows according to the vector fields
            flow = vector_fields.integrate(sampler_options=self.sampler_options)

            # push-forward template image
            forward_pushed_input = template.push_forward(flow)

            # compute the current energy consisting of intensity difference
            # and regularization
            energy_regularizer = self.regularizer.helmholtz(v0).get_norm(restriction=restriction)**2
            energy_intensity_unscaled = compute_energy(forward_pushed_input)
            energy_intensity = 1 / sigma**2 * energy_intensity_unscaled
            energy = energy_regularizer + energy_intensity

            if compute_grad:
                # compute gradient of the intensity difference
                gradient_l2_energy = compute_grad_energy(forward_pushed_input) / sigma**2

                # compute gradient of the intensity difference with respect
                # to the initial vector field
                gradient_initial_vector = self.integrate_backward_adjoint_Jacobi_field(gradient_l2_energy,
                                                                                       vector_fields)

                if return_all_energies:
                    return energy, energy_regularizer, energy_intensity_unscaled, energy_intensity, \
                            gradient_initial_vector.to_numpy().flatten()
                else:
                    return energy, gradient_initial_vector.to_numpy().flatten()
            else:
                if return_all_energies:
                    return energy, energy_regularizer, energy_intensity_unscaled, energy_intensity
                else:
                    return energy

        def save_current_state(x):
            opt['x'] = x

        # use scipy optimizer for minimizing energy function
        with self.logger.block("Perform image matching via geodesic shooting ..."):
            if optimization_method == 'GD':
                def gradient_descent(func, x0, grad_norm_tol=1e-5, rel_func_update_tol=1e-8, maxiter=1000,
                                     maxiter_armijo=20, alpha0=1., rho=0.5, c1=1e-4, disp=True, callback=None):
                    assert grad_norm_tol > 0 and rel_func_update_tol > 0
                    assert isinstance(maxiter, int) and maxiter > 0

                    def line_search(x, func_x, grad_x, d):
                        alpha = alpha0
                        d_dot_grad = d.dot(grad_x)
                        func_x_update = func(x + alpha * d, compute_grad=False)
                        k = 0
                        while (not func_x_update <= func_x + c1 * alpha * d_dot_grad) and k < maxiter_armijo:
                            alpha *= rho
                            func_x_update = func(x + alpha * d, compute_grad=False)
                            k += 1
                        return alpha

                    message = ''
                    with self.logger.block('Starting optimization using gradient descent ...'):
                        x = x0
                        if callback is not None:
                            callback(np.copy(x))
                        func_x, grad_x = func(x)
                        old_func_x = func_x
                        rel_func_update = rel_func_update_tol + 1
                        norm_grad_x = np.linalg.norm(grad_x)
                        i = 0
                        if disp:
                            self.logger.info(f'iter: {i:5d}\tf= {func_x:.5e}\t|grad|= {norm_grad_x:.5e}')
                        try:
                            while True:
                                if callback is not None:
                                    callback(np.copy(x))
                                if norm_grad_x <= grad_norm_tol:
                                    message = 'gradient norm below tolerance'
                                    break
                                elif rel_func_update <= rel_func_update_tol:
                                    message = 'relative function value update below tolerance'
                                    break
                                elif i >= maxiter:
                                    message = 'maximum number of iterations reached'
                                    break

                                d = -grad_x
                                alpha = line_search(x, func_x, grad_x, d)
                                x = x + alpha * d
                                func_x, grad_x = func(x)
                                if not np.isclose(old_func_x, 0.):
                                    rel_func_update = abs((func_x - old_func_x) / old_func_x)
                                else:
                                    rel_func_update = 0.
                                old_func_x = func_x
                                norm_grad_x = np.linalg.norm(grad_x)
                                i += 1
                                if disp:
                                    self.logger.info(f'iter: {i:5d}\tf= {func_x:.5e}\t|grad|= {norm_grad_x:.5e}')
                        except KeyboardInterrupt:
                            message = 'optimization stopped due to keyboard interrupt'
                            self.logger.warning('Optimization interrupted ...')

                    self.logger.info('Finished optimization ...')
                    result = {'x': x, 'nit': i, 'message': message}
                    return result

                res = gradient_descent(energy_and_gradient, initial_vector_field.to_numpy().flatten(),
                                       callback=save_current_state, **optimizer_options)
            else:
                save_current_state(initial_vector_field.to_numpy().flatten())
                res = optimize.minimize(energy_and_gradient, initial_vector_field.to_numpy().flatten(),
                                        method=optimization_method, jac=True, options=optimizer_options,
                                        callback=save_current_state)

        # compute time-dependent vector field from optimal initial vector field
        vector_fields = self.integrate_forward_vector_field(VectorField(data=res['x'].reshape((*self.shape, self.dim))))

        # compute forward flows according to the vector fields
        flow = vector_fields.integrate(sampler_options=self.sampler_options)

        # push-forward input-image
        transformed_input = template.push_forward(flow)

        opt['initial_vector_field'] = VectorField(data=res['x'].reshape((*self.shape, self.dim)))
        opt['transformed_input'] = transformed_input
        opt['flow'] = flow
        opt['vector_fields'] = vector_fields
        energy, energy_regularizer, energy_intensity_unscaled, energy_intensity, gradient = \
            energy_and_gradient(opt['x'], compute_grad=True, return_all_energies=True)
        opt['energy_regularizer'] = energy_regularizer
        opt['energy_intensity_unscaled'] = energy_intensity_unscaled
        opt['energy_intensity'] = energy_intensity
        opt['energy'] = energy
        opt['norm_gradient'] = np.linalg.norm(gradient)

        elapsed_time = int(time.perf_counter() - start_time)

        opt['reason_registration_ended'] = res['message']
        self.logger.info(f"Finished registration ({opt['reason_registration_ended']}) ...")

        if opt['initial_vector_field'] is not None:
            # compute the length of the path on the manifold;
            # this step only requires the initial vector due to conservation of momentum
            length = self.regularizer.cauchy_navier(opt['initial_vector_field']).get_norm(restriction=restriction)
        else:
            length = 0.0

        opt['length'] = length
        opt['iterations'] = res['nit']
        opt['time'] = elapsed_time
        opt['restriction'] = restriction

        if log_summary:
            self.summarize_results(opt, restriction=restriction)

        if return_all:
            return opt
        return opt['initial_vector_field']

    def summarize_results(self, results, restriction=np.s_[...]):
        """Log a summary of the results to the console.

        Parameters
        ----------
        results
            Dictionary with the results obtained from the `register`-function.
        restriction
            Slice that can be used to restrict the domain on which to compute the error.
        """
        self.logger.info("")
        self.logger.info("Registration summary")
        self.logger.info("====================")
        self.logger.info(f"Registration finished after {results['iterations']} iterations.")
        self.logger.info(f"Registration took {results['time']} seconds.")
        self.logger.info(f"Reason for the registration algorithm to stop: {results['reason_registration_ended']}.")
        norm_difference = (results['target'] - results['transformed_input']).get_norm(restriction=restriction)
        self.logger.info("Relative norm of difference: "
                         f"{norm_difference / results['target'].get_norm(restriction=restriction)}")
        self.logger.info("====================")

    def integrate_forward_vector_field(self, initial_vector_field):
        """Performs forward integration of the initial vector field according to EPDiff.

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
        `TimeDependentVectorField` obtained via forward integration of the initial `VectorField`.
        """
        if hasattr(self, 'shape'):
            assert self.shape == initial_vector_field.spatial_shape
            assert self.dim == initial_vector_field.dim
        else:
            self.shape = initial_vector_field.spatial_shape
            self.dim = initial_vector_field.dim
        # set up time-dependent vector field and set initial value
        vector_fields = TimeDependentVectorField(self.shape, self.time_steps)
        vector_fields[0] = initial_vector_field

        # einsum strings used for multiplication of (transposed) Jacobian matrix of vector fields
        einsum_string = '...lk,...k->...l'
        einsum_string_transpose = '...kl,...k->...l'

        def rhs_function(x):
            # compute the current momentum
            momentum_t = self.regularizer.cauchy_navier(x)
            # compute the gradient (Jacobian) of the current momentum
            grad_mt = momentum_t.grad
            # compute the divergence and the gradient (Jacobian) of the current vector field
            div_vt, grad_vt = x.get_divergence(return_gradient=True)
            # compute the right hand side, i.e. Dv^T m + Dm v + m div v
            rhs = (np.einsum(einsum_string_transpose, grad_vt, momentum_t.to_numpy())
                   + np.einsum(einsum_string, grad_mt, x.to_numpy())
                   + momentum_t.to_numpy() * div_vt[..., np.newaxis])
            rhs = VectorField(data=rhs)
            rhs = -self.regularizer.cauchy_navier_inverse(rhs)
            return rhs

        ti = self.time_integrator(rhs_function, self.dt)

        # perform forward in time integration of initial vector field
        for t in range(0, self.time_steps-1):
            # perform the explicit Euler integration step
            vector_fields[t+1] = ti.step(vector_fields[t])

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
        if hasattr(self, 'dim'):
            assert self.dim == vector_fields[0].dim
        else:
            self.dim = vector_fields[0].dim
        # introduce adjoint variables
        v_old = gradient_l2_energy
        delta_v = VectorField(v_old.spatial_shape)

        # einsum strings used for multiplication of (transposed) Jacobian matrix of vector fields
        einsum_string = '...lk,...l->...k'
        einsum_string_transpose = '...kl,...l->...k'

        def rhs_function(x, v):
            delta_v, v_old = x[0], x[1]
            # get divergence and gradient (Jacobian) of the current vector field
            div_vector_fields, grad_vector_fields = v.get_divergence(return_gradient=True)
            # get momentum corresponding to the adjoint variable `v_old`
            regularized_v = self.regularizer.cauchy_navier(v_old)
            # get gradient of the momentum of `v_old`
            grad_regularized_v = regularized_v.grad

            # update adjoint variable `v_old`
            rhs_v = - self.regularizer.cauchy_navier_inverse(
                VectorField(data=np.einsum(einsum_string_transpose, grad_vector_fields, regularized_v.to_numpy()))
                + VectorField(data=np.einsum(einsum_string, grad_regularized_v, v.to_numpy()))
                + regularized_v * div_vector_fields[..., np.newaxis])

            # get divergence and gradient of the adjoint variable `delta_v`
            div_delta_v, grad_delta_v = delta_v.get_divergence(return_gradient=True)
            # get momentum corresponding to the current vector field
            regularized_vector_fields = self.regularizer.cauchy_navier(v)
            # get gradient of the momentum of the current vector field
            grad_regularized_vector_fields = regularized_vector_fields.grad
            # update the adjoint variable `delta_v`
            rhs_delta_v = (- v_old
                           - (np.einsum(einsum_string, grad_vector_fields, delta_v.to_numpy())
                              - np.einsum(einsum_string, grad_delta_v, v.to_numpy()))
                           + self.regularizer.cauchy_navier_inverse(
                               VectorField(data=np.einsum(einsum_string_transpose,
                                                          grad_delta_v,
                                                          regularized_vector_fields.to_numpy()))
                               + VectorField(data=np.einsum(einsum_string,
                                                            grad_regularized_vector_fields,
                                                            delta_v.to_numpy()))
                               + regularized_vector_fields * div_delta_v[..., np.newaxis]))
            return rhs_delta_v, rhs_v

        ti = self.time_integrator(rhs_function, self.dt)

        # perform backward in time integration of the gradient of the energy function
        for t in range(self.time_steps-2, -1, -1):
            delta_v, v_old = ti.step_backwards([delta_v, v_old], {'v': vector_fields[t]})

        # return adjoint variable `delta_v` that corresponds to the gradient
        # of the objective function at the initial time instance
        return delta_v

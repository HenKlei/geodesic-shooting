import numpy as np

from geodesic_shooting.utils.logger import getLogger


def gradient_descent(func, x0, grad_norm_tol=1e-5, rel_func_update_tol=1e-8, maxiter=1000,
                     maxiter_armijo=20, alpha0=1., rho=0.5, c1=1e-4, logger=None, disp=True, callback=None):
    assert grad_norm_tol > 0 and rel_func_update_tol > 0
    assert isinstance(maxiter, int) and maxiter > 0

    if logger is None:
        logger = getLogger('gradient_descent', level='INFO')

    def line_search(x, func_x, grad_x, d):
        alpha = alpha0
        d_dot_grad = d.flatten().dot(grad_x.flatten())
        func_x_update = func(x + alpha * d, compute_grad=False)
        k = 0
        while (not func_x_update <= func_x + c1 * alpha * d_dot_grad) and k < maxiter_armijo:
            alpha *= rho
            func_x_update = func(x + alpha * d, compute_grad=False)
            k += 1
        return alpha

    message = ''
    with logger.block('Starting optimization using gradient descent ...'):
        x = x0
        if callback is not None:
            callback(np.copy(x))
        func_x, grad_x = func(x)
        old_func_x = func_x
        rel_func_update = rel_func_update_tol + 1
        norm_grad_x = np.linalg.norm(grad_x)
        i = 0
        if disp:
            logger.info(f'iter: {i:5d}\tf= {func_x:.5e}\t|grad|= {norm_grad_x:.5e}')
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
                    logger.info(f'iter: {i:5d}\tf= {func_x:.5e}\t|grad|= {norm_grad_x:.5e}')
        except KeyboardInterrupt:
            message = 'optimization stopped due to keyboard interrupt'
            logger.warning('Optimization interrupted ...')

    logger.info('Finished optimization ...')
    result = {'x': x, 'nit': i, 'message': message}
    return result
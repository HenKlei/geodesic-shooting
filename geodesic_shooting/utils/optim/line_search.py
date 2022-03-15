import numpy as np

from geodesic_shooting.utils.logger import getLogger


class BaseLineSearch:
    """Base class for the line search algorithms.

    A certain line search algorithm is specified by writing the `__call__` method.
    The `BaseLineSearch` class implements no line search, but performs an optimization
    step with fixed stepsize 1.
    """
    def __init__(self, energy_func, gradient_func):
        """Constructor.

        Parameters
        ----------
        energy_func
            Objective function that is to be minimized.
        gradient_func
            Function that computes the gradient of `energy_func` at a given point.
        """
        self.energy_func = energy_func
        self.gradient_func = gradient_func

    def __call__(self, optimizer, x, energy, grad, min_stepsize=1e-4, max_stepsize=1., max_num_search_steps=20):
        """Function that performs the line search.

        Parameters
        ----------
        optimizer
            Optimizer that is used for optimization (required to perform optimization steps
            for fixed stepsizes).
        x
            Current iterate of the optimization algorithm.
        energy
            Value of the objective function at `x`.
        grad
            Gradient of the objective function at `x`.
        min_stepsize
            Smallest possible stepsize.
        max_stepsize
            Largest possible stepsize.
        max_num_search_steps
            Maximum number of search steps (usually the maximum number of function evaluations
            during line search).

        Returns
        -------
        Stepsize, updated iterate, energy at the new position,
        and the gradient at the new position.
        """
        epsilon = 1.
        new_x = optimizer.step_given_stepsize(x, epsilon, grad)

        new_energy, additional_infos = self.energy_func(new_x, return_additional_infos=True)

        new_grad = self.gradient_func(new_x, **additional_infos)

        return epsilon, new_x, new_energy, new_grad


class ArmijoLineSearch(BaseLineSearch):
    """Class that implements an Armijo line search scheme.

    Beginning with a maximum stepsize, the stepsize is successively reduced by a certain
    factor until a stepsize is reached such that the resulting point leads to a smaller
    objective function value.
    """
    def __init__(self, energy_func, gradient_func, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        energy_func
            Objective function that is to be minimized.
        gradient_func
            Function that computes the gradient of `energy_func` at a given point.
        log_level
            Level of the log messages to display (required by the logger).
        """
        self.energy_func = energy_func
        self.gradient_func = gradient_func

        self.logger = getLogger('armijo_line_search', level=log_level)

    def __call__(self, optimizer, x, energy, grad, min_stepsize=1e-4, max_stepsize=1., max_num_search_steps=20):
        """Function that performs the line search.

        Parameters
        ----------
        optimizer
            Optimizer that is used for optimization (required to perform optimization steps
            for fixed stepsizes).
        x
            Current iterate of the optimization algorithm.
        energy
            Value of the objective function at `x`.
        grad
            Gradient of the objective function at `x`.
        min_stepsize
            Smallest possible stepsize.
        max_stepsize
            Largest possible stepsize.
        max_num_search_steps
            Maximum number of search steps (usually the maximum number of function evaluations
            during line search).

        Returns
        -------
        Stepsize, updated iterate, energy at the new position,
        and the gradient at the new position.
        """
        assert min_stepsize <= max_stepsize
        assert max_num_search_steps >= 1

        epsilon = max_stepsize
        rho = pow(min_stepsize / max_stepsize, 1. / max_num_search_steps)
        epsilon = max_stepsize
        reducing_stepsize = False

        new_x = optimizer.step_given_stepsize(x, epsilon, grad)
        new_energy, additional_infos = self.energy_func(new_x, return_additional_infos=True)
        while not np.isclose(rho, 1.) and epsilon * rho >= min_stepsize and new_energy >= energy:
            epsilon *= rho
            reducing_stepsize = True
            new_x = optimizer.step_given_stepsize(x, epsilon, grad)
            new_energy, additional_infos = self.energy_func(new_x, return_additional_infos=True)

        if reducing_stepsize:
            self.logger.info(f'Reducing stepsize to {epsilon:.3e} ...')

        new_grad = self.gradient_func(new_x, **additional_infos)

        return epsilon, new_x, new_energy, new_grad

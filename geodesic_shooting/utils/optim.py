import inspect
import numpy as np

from geodesic_shooting.utils.logger import getLogger


class BaseOptimizer:
    """Base class for the optimization algorithms.

    A certain optimization algorithm is specified by writing the `step_given_stepsize` method.
    """
    def __init__(self, line_search):
        """Constructor.

        Parameters
        ----------
        line_search
            Instance of line search algorithm (knows about the objective function
            and its gradient). Can also be an instance of `BaseLineSearch`,
            then no line search is performed.
        """
        self.line_search = line_search

    def step(self, x, energy, grad,
             parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1.,
                                     'max_num_search_steps': 10}):
        """Performs a single optimization step.

        Parameters
        ----------
        x
            Current iterate of the optimization procedure.
        energy
            Current value of the energy/objective function.
        grad
            Current gradient of the objective function.
        parameters_line_search
            Additional parameters passed to the line search algorithm (can be used for instance to
            adaptively adjust the maximum stepsize during optimization).

        Returns
        -------
        The new iterate, energy, gradient and the stepsize used for the optimization step.
        """
        stepsize, new_x, new_energy, new_grad = self.line_search(self, x, energy, grad, **parameters_line_search)
        return new_x, new_energy, new_grad, stepsize

    def step_given_stepsize(self, x, stepsize, grad):
        raise NotImplementedError


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


class BaseStepsizeController:
    """Base class for methods to control stepsizes and other line search parameters.

    These algorithms adjust parameters like minimal or maximal stepsize before performing the next
    optimization step. Furthermore, additional parameters of the line search algorithm can also be
    changed using these algorithms.
    """
    def __init__(self, line_search, log_frequency=10, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        line_search
            Line search algorithm that is used in the optimization.
        log_frequency
            Frequency of logging the stepsize.
        log_level
            Level of the log messages to display (required by the logger).
        """
        self.line_search = line_search

        self.log_frequency = log_frequency

        self.reset()

        self.logger = getLogger('stepsize_controller', level=log_level)

    def reset(self):
        """Function to reset internal parameters of the stepsize controller."""
        self.number_iterations_without_logging = 0

    def _check_parameters(self, parameters_line_search):
        """Function to check whether the parameters given to the stepsize controller can fit the
        parameters used in the line search algorithm.

        Parameters
        ----------
        parameters_line_search
            Dictionary provided for updating the stepsize and that is supposed to be checked
            whether it can be used by the line search algorithm.
        """
        args = inspect.getfullargspec(self.line_search.__call__)[0]
        assert all(param in args for param in parameters_line_search)

    def update(self, parameters_line_search, current_stepsize):
        """Function that updates the minimal and maximal stepsize or other parameters of the line
        search algorithm.

        This function should always first check if the parameters fit to the line search algorithm,
        i.e. it should first call `_check_parameters`.

        Parameters
        ----------
        parameters_line_search
            Dictionary with additional information for the line search algorithm that is to be
            updated.
        current_stepsize
            The last stepsize that was returned by the line search algorithm.

        Returns
        -------
        A dictionary with the adjusted parameters.
        """
        self._check_parameters(parameters_line_search)
        assert 'max_stepsize' in parameters_line_search
        assert 'min_stepsize' in parameters_line_search

        old_max_stepsize = parameters_line_search['max_stepsize']
        old_min_stepsize = parameters_line_search['min_stepsize']

        parameters_line_search['max_stepsize'] = min(parameters_line_search['max_stepsize'],
                                                     current_stepsize)
        parameters_line_search['min_stepsize'] = min(parameters_line_search['min_stepsize'],
                                                     parameters_line_search['max_stepsize'])

        if not np.isclose(old_max_stepsize, parameters_line_search['max_stepsize']):
            self.logger.info(f'Updating maximum stepsize to {parameters_line_search["max_stepsize"]:.3e} ...')

        if not np.isclose(old_min_stepsize, parameters_line_search['min_stepsize']):
            self.logger.info(f'Updating minimum stepsize to {parameters_line_search["min_stepsize"]:.3e} ...')

        if self.number_iterations_without_logging >= self.log_frequency:
            self.logger.info(f'Current maximum stepsize: {parameters_line_search["max_stepsize"]:.3e}')
            self.logger.info(f'Current minimum stepsize: {parameters_line_search["min_stepsize"]:.3e}')
            self.number_iterations_without_logging = 0

        self.number_iterations_without_logging += 1

        return parameters_line_search


class GradientDescentOptimizer(BaseOptimizer):
    """Class that implements a simple gradient descent optimization algorithm.

    The algorithm can be combined with a line search algorithm since it inherits
    the `step` method from `BaseOptimizer`.
    """
    def step_given_stepsize(self, x, stepsize, grad):
        """Performs a single optimization step for a given stepsize.

        Simple gradient descent step, i.e. a step with fixed stepsize in the direction
        of the negative gradient.

        Parameters
        ----------
        x
            Current iterate of the optimizer.
        stepsize
            Stepsize to use for the optimization step.
        grad
            Gradient of the objective function at `x`.

        Returns
        -------
        Updated position after applying gradient descent step.
        """
        return x - stepsize * grad


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
        rho = pow(min_stepsize / max_stepsize, 1./max_num_search_steps)
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


class PatientStepsizeController(BaseStepsizeController):
    """Stepsize controller that waits some iterations before changing stepsize.

    A fixed number of iterations with reduced stepsize before finally adjusting minimal
    and maximal stepsize can be prescribed. The new maximal stepsize will be set to the
    maximal stepsize that occured during waiting before adjusting the stepsizes.
    """
    def __init__(self, line_search, patience=10, log_frequency=10, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        line_search
            Line search algorithm that is used in the optimization.
        patience
            Number of iterations to wait before changing stepsize.
        log_frequency
            Frequency of logging the stepsize.
        log_level
            Level of the log messages to display (required by the logger).
        """
        self.line_search = line_search

        self.patience = patience

        self.log_frequency = log_frequency

        self.reset()

        self.logger = getLogger('stepsize_controller', level=log_level)

    def reset(self):
        """Function to reset internal parameters of the stepsize controller."""
        self.maximal_reduced_stepsize = None
        self.number_iterations_without_changing_stepsize = 0
        self.number_iterations_without_logging = 0

    def update(self, parameters_line_search, current_stepsize):
        """Function that updates the minimal and maximal stepsize of the line search algorithm.

        Whether or not to update the stepsize depends on the number of consecutive iterations in
        which the stepsize was reduced.

        Parameters
        ----------
        parameters_line_search
            Dictionary with additional information for the line search algorithm that is to be
            updated.
        current_stepsize
            The last stepsize that was returned by the line search algorithm.

        Returns
        -------
        A dictionary with the adjusted parameters.
        """
        self._check_parameters(parameters_line_search)
        assert 'max_stepsize' in parameters_line_search
        assert 'min_stepsize' in parameters_line_search

        old_max_stepsize = parameters_line_search['max_stepsize']
        old_min_stepsize = parameters_line_search['min_stepsize']

        if not np.isclose(parameters_line_search['max_stepsize'], current_stepsize):
            if self.number_iterations_without_changing_stepsize < self.patience:
                if self.maximal_reduced_stepsize is None:
                    self.maximal_reduced_stepsize = current_stepsize
                else:
                    self.maximal_reduced_stepsize = max(self.maximal_reduced_stepsize, current_stepsize)
                self.number_iterations_without_changing_stepsize += 1
            else:
                parameters_line_search['max_stepsize'] = self.maximal_reduced_stepsize
                assert not np.isclose(parameters_line_search['max_stepsize'], old_max_stepsize)
                self.logger.info(f'Updating maximum stepsize to {parameters_line_search["max_stepsize"]:.3e} ...')

                parameters_line_search['min_stepsize'] = min(parameters_line_search['min_stepsize'],
                                                             parameters_line_search['max_stepsize'])
                if not np.isclose(old_min_stepsize, parameters_line_search['min_stepsize']):
                    self.logger.info(f'Updating minimum stepsize to {parameters_line_search["min_stepsize"]:.3e} ...')

                self.number_iterations_without_changing_stepsize = 0
                self.maximal_reduced_stepsize = None
        else:
            self.number_iterations_without_changing_stepsize = 0
            self.maximal_reduced_stepsize = None

        if self.number_iterations_without_logging >= self.log_frequency:
            self.logger.info(f'Current maximum stepsize: {parameters_line_search["max_stepsize"]:.3e}')
            self.logger.info(f'Current minimum stepsize: {parameters_line_search["min_stepsize"]:.3e}')
            self.number_iterations_without_logging = 0

        self.number_iterations_without_logging += 1

        return parameters_line_search

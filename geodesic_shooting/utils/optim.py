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

        new_energy, velocity_fields, forward_pushed_input = self.energy_func(new_x)

        new_grad = self.gradient_func(new_x, velocity_fields, forward_pushed_input)

        return epsilon, new_x, new_energy, new_grad


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
        epsilon = max_stepsize
        rho = pow(min_stepsize / max_stepsize, 1./max_num_search_steps)
        epsilon = max_stepsize
        reducing_stepsize = False

        new_x = optimizer.step_given_stepsize(x, epsilon, grad)
        new_energy, velocity_fields, forward_pushed_input = self.energy_func(new_x)
        while epsilon > min_stepsize and new_energy >= energy:
            epsilon *= rho
            reducing_stepsize = True
            new_x = optimizer.step_given_stepsize(x, epsilon, grad)
            new_energy, velocity_fields, forward_pushed_input = self.energy_func(new_x)

        if reducing_stepsize:
            self.logger.info(f'Reducing stepsize to {epsilon:.3e} ...')

        new_grad = self.gradient_func(new_x, velocity_fields, forward_pushed_input)

        return epsilon, new_x, new_energy, new_grad

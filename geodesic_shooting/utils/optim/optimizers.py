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

from geodesic_shooting.utils.logger import getLogger


class BaseOptimizer:
    def step(self, x, energy, grad):
        raise NotImplementedError


class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self, line_search):
        self.line_search = line_search

    def step(self, x, energy, grad,
             parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1.,
                                     'max_num_search_steps': 10}):
        stepsize, x, new_energy, new_grad = self.line_search(self, x, energy, grad, **parameters_line_search)
        return x, new_energy, new_grad, stepsize

    def step_given_stepsize(self, x, stepsize, grad):
        return x - stepsize * grad


class ArmijoLineSearch:
    def __init__(self, energy_func, gradient_func, log_level='INFO'):
        self.energy_func = energy_func
        self.gradient_func = gradient_func

        self.logger = getLogger('armijo_line_search', level=log_level)

    def __call__(self, optimizer, x, energy, grad, min_stepsize=1e-4, max_stepsize=1., max_num_search_steps=20):
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

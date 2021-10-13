from geodesic_shooting.utils.logger import getLogger


class BaseOptimizer:
    def step(self, x, energy, grad):
        raise NotImplementedError


class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self, line_search, energy_and_gradient):
        self.line_search = line_search
        self.energy_and_gradient = energy_and_gradient

    def step(self, x, energy, grad,
             parameters_line_search={'min_stepsize': 1e-4, 'max_stepsize': 1.,
                                     'max_num_search_steps': 10}):
        stepsize = self.line_search(self, x, energy, grad, **parameters_line_search)
        x = self.step_given_stepsize(x, stepsize, grad)
        new_energy, new_grad = self.energy_and_gradient(x)
        return x, new_energy, new_grad, stepsize

    def step_given_stepsize(self, x, stepsize, grad):
        return x - stepsize * grad


class ArmijoLineSearch:
    def __init__(self, func, log_level='INFO'):
        self.func = func

        self.logger = getLogger('armijo_line_search', level=log_level)

    def __call__(self, optimizer, x, energy, grad, min_stepsize=1e-4, max_stepsize=1., max_num_search_steps=20):
        epsilon = max_stepsize
        rho = pow(min_stepsize / max_stepsize, 1./max_num_search_steps)
        epsilon = max_stepsize
        reducing_stepsize = False

        while epsilon > min_stepsize and self.func(optimizer.step_given_stepsize(x, epsilon, grad))[0] >= energy:
            epsilon *= rho
            reducing_stepsize = True

        if reducing_stepsize:
            self.logger.info(f'Reducing stepsize to {epsilon:.3e} ...')

        return epsilon

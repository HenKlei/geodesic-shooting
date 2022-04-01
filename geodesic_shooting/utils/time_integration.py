

class TimeIntegrator():
    def __init__(self, f, dt):
        self.f = f
        self.dt = dt

    def step(self, x, additional_args={}):
        raise NotImplementedError

    def step_backwards(self, x, additional_args={}):
        raise NotImplementedError


class ExplicitEuler(TimeIntegrator):
    def step(self, x, additional_args={}):
        return x + self.dt * self.f(x, **additional_args)

    def step_backwards(self, x, additional_args={}):
        return x - self.dt * self.f(x, **additional_args)


class RK4(TimeIntegrator):
    def _rhs(self, x, additional_args={}):
        k1 = self.f(x, **additional_args)
        k2 = self.f(x + self.dt/2. * k1, **additional_args)
        k3 = self.f(x + self.dt/2. * k2, **additional_args)
        k4 = self.f(x + self.dt * k3, **additional_args)
        return self.dt/6. * (k1 + 2.*k2 + 2.*k3 + k4)

    def step(self, x, additional_args={}):
        return x + self._rhs(x, additional_args=additional_args)

    def step_backwards(self, x, additional_args={}):
        return x - self._rhs(x, additional_args=additional_args)

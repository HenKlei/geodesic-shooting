from abc import abstractmethod


class TimeIntegrator():
    def __init__(self, f, dt):
        self.f = f
        self.dt = dt

    @abstractmethod
    def _rhs(self, x, additional_args={}, sign=1):
        pass

    def step(self, x, additional_args={}):
        if isinstance(x, list):
            return [y + self.dt * r for y, r in zip(x, self._rhs(x, additional_args=additional_args, sign=1))]
        return x + self.dt * self._rhs(x, additional_args=additional_args, sign=1)

    def step_backwards(self, x, additional_args={}):
        if isinstance(x, list):
            return [y - self.dt * r for y, r in zip(x, self._rhs(x, additional_args=additional_args, sign=-1))]
        return x - self.dt * self._rhs(x, additional_args=additional_args, sign=-1)


class ExplicitEuler(TimeIntegrator):
    def _rhs(self, x, additional_args={}, sign=1):
        return self.f(x, **additional_args)


class RK4(TimeIntegrator):
    def _rhs(self, x, additional_args={}, sign=1):
        k1 = self.f(x, **additional_args)
        if isinstance(x, list):
            temp = [y + sign * self.dt/2. * r for y, r in zip(x, k1)]
        else:
            temp = x + sign * self.dt/2. * k1
        k2 = self.f(temp, **additional_args)
        if isinstance(x, list):
            temp = [y + sign * self.dt/2. * r for y, r in zip(x, k2)]
        else:
            temp = x + sign * self.dt/2. * k2
        k3 = self.f(temp, **additional_args)
        if isinstance(x, list):
            temp = [y + sign * self.dt * r for y, r in zip(x, k3)]
        else:
            temp = x + sign * self.dt * k3
        k4 = self.f(temp, **additional_args)
        if isinstance(x, list):
            return [(l1 + 2.*l2 + 2.*l3 + l4) / 6. for l1, l2, l3, l4 in zip(k1, k2, k3, k4)]
        return (k1 + 2.*k2 + 2.*k3 + k4) / 6.

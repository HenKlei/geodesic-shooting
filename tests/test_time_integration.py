import numpy as np

from geodesic_shooting.utils.time_integration import RK4, ExplicitEuler


def test_RK4():
    def f(x):
        return - x - 2.

    def exact_solution(t):
        return 2. * (np.exp(2. - t) - 1.)

    time_steps = 100
    dt = 2. / time_steps

    time_integrator = RK4(f, dt)

    x = exact_solution(0)
    for t in range(time_steps, 0, -1):
        x = time_integrator.step(x)

    assert np.abs(x - exact_solution(2)) < 1e-8

    x = 0.
    for t in range(time_steps, 0, -1):
        x = time_integrator.step_backwards(x)

    assert np.abs(x - exact_solution(0)) < 1e-6


def test_explicit_Euler():
    def f(x):
        return - x - 2.

    def exact_solution(t):
        return 2. * (np.exp(2. - t) - 1.)

    time_steps = 10000
    dt = 2. / time_steps

    time_integrator = ExplicitEuler(f, dt)

    x = exact_solution(0)
    for t in range(time_steps, 0, -1):
        x = time_integrator.step(x)

    assert np.abs(x - exact_solution(2)) < 1e-3

    x = 0.
    for t in range(time_steps, 0, -1):
        x = time_integrator.step_backwards(x)

    assert np.abs(x - exact_solution(0)) < 1e-2

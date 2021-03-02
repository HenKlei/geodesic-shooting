import numpy as np

from pymor.basic import *

import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid


N = 50

nt = 100

T = 0.25

dt = T / nt

dirichlet_data = ConstantFunction(dim_domain=2, value=0.)
initial_data = ExpressionFunction("(x[..., 0] >= 0.25) * (x[..., 0] <= 0.5) * (x[..., 1] >= 0.25) * (x[..., 1] <= 0.5) * 1", 2, ())

vx = 1.
vy = 1.

parameter_range = (1., 2.)

problem = InstationaryProblem(
        StationaryProblem(
            domain=RectDomain(right=None, top=None),
            dirichlet_data=dirichlet_data,
            rhs=None,
            advection=LincombFunction([ConstantFunction(np.array([vx, vy]), dim_domain=2)],
                                      [ProjectionParameterFunctional('speed')]),
        ),
        initial_data=initial_data,
        T=T,
        parameter_ranges=parameter_range,
        name="transport_problem"
    )

fom, _ = discretize_instationary_fv(problem, diameter=1./N, grid_type=RectGrid, nt=nt)

mu = fom.parameters.parse({'speed': 1.0})

u_orig = fom.solve(mu)
fom.visualize(u_orig)


def solution_to_image(u):
    u_transformed = np.reshape(u.to_numpy(), (71, 71))
    return u_transformed

def image_to_solution(u):
    u = u.ravel()
    return fom.operator.source.from_numpy([u,])


x_transformed = np.linspace(0., 1., 71)
y_transformed = np.linspace(0., 1., 71)

import matplotlib.pyplot as plt
fig, axes = plt.subplots()
cf = axes.contourf(x_transformed, y_transformed, solution_to_image(image_to_solution(solution_to_image(u_orig[0]))), 100, cmap='jet')
axes.axis('equal')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots()
cf = axes.contourf(x_transformed, y_transformed, solution_to_image(u_orig[-1]), 100, cmap='jet')
axes.axis('equal')
plt.tight_layout()
plt.show()

lddmm = pyLDDMM.LDDMM()

implicit_Euler = False#True

u_num = [u_orig[0],]

for n in range(nt):
    problem = pyLDDMM.PDEProblemInstationary(fom, solution_to_image(u_num[-1]), solution_to_image, image_to_solution, dt=dt, implicit_Euler=implicit_Euler, alpha=1, gamma=1)
    image, v, energies, Phi0, Phi1, J0, length = lddmm.register(solution_to_image(u_num[-1]), problem, K=500, sigma=0.05, epsilon=0.000005, mu=mu, return_all=True)
    u_num.append(image_to_solution(image))

    '''
    fig, axes = plt.subplots()
    cf = axes.contourf(x_transformed, y_transformed, solution_to_image(u_num[-1]), 100, cmap='jet')
    axes.axis('equal')
    plt.tight_layout()
    plt.show()
    '''

fig, axes = plt.subplots()
cf = axes.contourf(x_transformed, y_transformed, solution_to_image(u_num[-1]), 100, cmap='jet')
axes.axis('equal')
plt.tight_layout()
plt.show()

print((u_orig[-1]-u_num[-1]).norm() / u_orig[-1].norm())

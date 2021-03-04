import numpy as np

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.models.basic import InstationaryModel
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction
from pymor.discretizers.builtin import discretize_stationary_cg, RectGrid
from pymor.bindings.fenics import (FenicsVectorSpace, FenicsOperator,
                                   FenicsVisualizer, FenicsMatrixOperator)
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.operators.constructions import VectorOperator

import dolfin as df

import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid


N_x = 50#100
N_y = 50#100

Nt = 10#0

# define extend of the rectangular computational domain
x_left = 0.0
x_right = 1.0
y_bottom = 0.0
y_top = 1.0
# get mesh for the domain
mesh = df.RectangleMesh(df.Point(x_left, y_bottom), df.Point(x_right, y_top), N_x, N_y)

# get function space for the mesh
V = df.FunctionSpace(mesh, 'CG', 1)

# define boundary conditions
u_left_bottom = df.Constant(0.0)

def on_left_bottom(x, on_boundary):
    return on_boundary and (df.near(x[0], x_left) or df.near(x[1], y_bottom))

bc = df.DirichletBC(V, u_left_bottom, on_left_bottom)

# define initial conditions
u_init = df.Expression("(x[0]>=0.25)*(x[0]<=0.5)*(x[1]>=0.25)*(x[1]<=0.5)", degree=1)#,2)
u_0 = df.project(u_init, V).vector()

# get trial and test functions
u = df.Function(V)
u_trial = df.TrialFunction(V)
v = df.TestFunction(V)

# define transport vector
b = df.Constant((1.0, 1.0))

speed = df.Constant(1.0)

# define equation in weak form
F = speed * df.dot(df.grad(u), b) * v * df.dx

mass_form = u_trial * v * df.dx
mass_matrix = df.assemble(mass_form)

# get fenics operator for the equation
op = FenicsOperator(F, FenicsVectorSpace(V), FenicsVectorSpace(V), u, (bc,),
                    parameter_setter=lambda mu: speed.assign(mu['speed'].item()),
                    parameters={'speed': 1,},
                    solver_options={'inverse': {'type': 'newton', 'rtol': 1e-6}})

# define zero vector for right hand side (reaction/source term is already part of the weak formulation)
rhs = VectorOperator(op.range.zeros())

#define final time
T = 0.25

# choose time stepping scheme
time_stepper = ImplicitEulerTimeStepper(Nt)

# define full-order model
fom = InstationaryModel(T,
                        FenicsVectorSpace(V).make_array([u_0]),
                        op,
                        rhs,
                        mass=FenicsMatrixOperator(mass_matrix, V, V, name='l2'),
                        time_stepper=time_stepper,
                        visualizer=FenicsVisualizer(FenicsVectorSpace(V)))


mu = fom.parameters.parse({'speed': 1.0})

u_orig = fom.solve(mu)

mesh = fom.visualizer.space.V.mesh()
n = fom.visualizer.space.V.dim()
d = mesh.geometry().dim()

dof_coordinates = fom.visualizer.space.V.tabulate_dof_coordinates()
dof_coordinates.resize((n, d))

dof_x = dof_coordinates[:, 0]
dof_y = dof_coordinates[:, 1]

dofs_sorted = sorted(zip(dof_x, dof_y))
list_zip_dofs = np.array(list(zip(dof_x, dof_y)))

grid = np.reshape(dofs_sorted, (N_x+1, -1, d))

x_transformed = np.array([[val[0] for val in r] for r in grid])
y_transformed = np.array([[val[1] for val in r] for r in grid])


def solution_to_image(u):
    _, _, u_sorted = zip(*sorted(zip(dof_x, dof_y, u.to_numpy().squeeze())))
    u_transformed = np.reshape(u_sorted, (N_x+1, -1))
    return u_transformed

def image_to_solution(u):
    u = u.ravel()
  
    res = np.zeros(u.shape[0])
    for t in zip(dofs_sorted, u):
        temp = np.array(list_zip_dofs == t[0]).T
        res[np.logical_and(temp[0], temp[1])] = t[1]

    return fom.operator.source.from_numpy([res,])


start_image = np.zeros((N_x+1, N_y+1))
start_image[int(N_x*0.1):int(N_x*0.35), int(N_y*0.1):int(N_y*0.35)] = 1



import matplotlib.pyplot as plt
fig, axes = plt.subplots()
cf = axes.contourf(x_transformed, y_transformed, solution_to_image(image_to_solution(solution_to_image(u_orig[1]))), 100, cmap='jet')
axes.axis('equal')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots()
cf = axes.contourf(x_transformed, y_transformed, solution_to_image(u_orig[-1]), 100, cmap='jet')
axes.axis('equal')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots()
cf = axes.contourf(x_transformed, y_transformed, start_image, 100, cmap='jet')
axes.axis('equal')
plt.tight_layout()
plt.show()


implicit_Euler = False#True

problem = pyLDDMM.PDEProblemInstationary(fom, solution_to_image(u_orig[1]), solution_to_image, image_to_solution, implicit_Euler=implicit_Euler, alpha=1, gamma=1)

# perform the registration
lddmm = pyLDDMM.LDDMM()
image, v, energies, Phi0, Phi1, J0, length = lddmm.register(solution_to_image(u_orig[1]), problem, K=500, sigma=0.05, epsilon=0.0001, mu=mu, return_all=True)


fig, axes = plt.subplots()
cf = axes.contourf(x_transformed, y_transformed, image, 100, cmap='jet')
axes.axis('equal')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots()
cf = axes.contourf(x_transformed, y_transformed, image-start_image, 100, cmap='jet')
axes.axis('equal')
plt.tight_layout()
plt.show()

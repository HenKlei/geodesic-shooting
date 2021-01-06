import numpy as np

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction
from pymor.discretizers.builtin import discretize_stationary_cg, RectGrid

import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid


if __name__ == "__main__":
    i0 = loadimg('./example_images/circle.png')
    i1 = loadimg('./example_images/square.png')

    problem = pyLDDMM.ImageRegistrationProblem(i1, alpha=1, gamma=1)

    # perform the registration
    lddmm_circle = pyLDDMM.LDDMM()
    im, v, energies, Phi0, Phi1, J0, J1, length = lddmm_circle.register(i0, problem, sigma=0.1, epsilon=0.0001, K=20, return_all=True)

    rhs = ExpressionFunction('ones(x.shape[:-1]) * 10', 2, ())
    dirichlet = ExpressionFunction('zeros(x.shape[:-1])', 2, ())
    domain = RectDomain()

    n = 63

    problem = StationaryProblem(
        domain=domain,
        diffusion=ConstantFunction(1, dim_domain=2),
        rhs=rhs,
        dirichlet_data=dirichlet,
    )

    fom, _ = discretize_stationary_cg(
        analytical_problem=problem,
        grid_type=RectGrid,
        diameter=np.sqrt(2) / n
    )

    problem = pyLDDMM.PDEProblemStationary(fom, alpha=1, gamma=1)

    mu = None

    u_orig = problem.fom.solve()

    u_transformed = lddmm_circle.push_forward(u_orig.to_numpy().reshape((n+1, n+1)), Phi0)[-1]
    problem.fom.visualize(problem.fom.operator.source.from_numpy(u_transformed.ravel()), title='Transformed solution')

#    u = lddmm_circle.pull_back(u_transformed, Phi1)[0]
#    problem.fom.visualize(problem.fom.operator.source.from_numpy(u_transformed.ravel()), title='Solution transformed back')
#    problem.fom.visualize(u_orig - problem.fom.operator.source.from_numpy(u_transformed.ravel()), title='Difference of transformed and inverse transformed')

    # perform the registration
    lddmm = pyLDDMM.LDDMM()
    image, v, energies, Phi0, Phi1, J0, length = lddmm.register(u_transformed, problem, K=500, sigma=0.01, epsilon=0.00001, mu=mu, return_all=True)

    image = problem.fom.operator.source.from_numpy(image.ravel())
    difference = image - u_orig
    problem.fom.visualize(difference, title='Difference')
    problem.fom.visualize(image, title='Solution via LDDMM')

    problem.fom.visualize(image - problem.fom.operator.source.from_numpy(u_transformed.ravel()), title='')

    print(f'Norm of difference: {difference.norm()}')
    print(f'Relative norm of difference: {difference.norm() / u_orig.norm()}')

    # save animation of the transformation
    save_animation('./example_images/out_pde.gif', J0)

    # plot the transformation
    plt = plot_warpgrid(Phi1[0], interval=1)
    plt.savefig('./example_images/out_pde_warp.png')

    square_inverse_transformed = lddmm.push_forward(i1, Phi0)[-1]
    saveimg('./example_images/square_inverse_transformed.png', square_inverse_transformed)

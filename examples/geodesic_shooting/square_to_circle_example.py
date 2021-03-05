import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid, plot_vector_field

if __name__ == "__main__":
    # load greyscale images
    i0 = loadimg('../example_images/circle.png')
    i1 = loadimg('../example_images/square.png')

    problem = pyLDDMM.ImageRegistrationProblemGS(i0, alpha=6, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.GeodesicShooting()
    im, v0, energies, Phi0, length = lddmm.register(i1, problem, sigma=0.1, epsilon=0.01, K=50, return_all=True)

    # save i0 aligned to i1
    saveimg('../example_images/out_s2c.png', im)

    plt_inverse = plot_warpgrid(Phi0, interval=2)
    plt_inverse.savefig('../example_images/out_s2c_warp_inverse.png')

    Phi_half = lddmm.integrate_forward_flow(lddmm.integrate_forward_vector_field(v0/2))
    saveimg('../example_images/out_s2c_half_speed.png', lddmm.push_forward(i0, Phi_half))

    # plot the (initial) vector field
    plt = plot_vector_field(v0, interval=2)
    plt.show()

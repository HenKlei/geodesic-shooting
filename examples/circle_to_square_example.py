import pyLDDMM
from pyLDDMM.utils.visualization import loadimg, saveimg, save_animation, plot_warpgrid

if __name__ == "__main__":
    # load greyscale images
    i0 = loadimg('./example_images/circle.png')
    i1 = loadimg('./example_images/square.png')

    problem = pyLDDMM.ImageRegistrationProblem(i1, alpha=1, gamma=1)

    # perform the registration
    lddmm = pyLDDMM.LDDMM()
    im, v, energies, Phi0, Phi1, J0, J1, length = lddmm.register(i0, problem, sigma=0.1, epsilon=0.0001, K=20, return_all=True)

    # save i0 aligned to i1
    saveimg('example_images/out_c2s.png', im)

    # save animation of the transformation
    save_animation('example_images/out_c2s.gif', J0)

    # plot the transfomration
    plt = plot_warpgrid(Phi1[0], interval=1)
    plt.savefig('example_images/out_c2s_warp.png')

import os
import matplotlib.pyplot as plt

from geodesic_shooting.utils.reduced import pod


def plot_registration_results(results, interval=1, frequency=1, scale=None, figsize=(20, 20),
                              show_restriction_boundary=True):
    """Plots some of the results from registration via geodesic shooting.

    Parameters
    ----------
    results
        Dictionary containing the results obtained via geodesic shooting.
    interval
        Interval in which to sample.
    frequency
        Frequency in which to sample the points in the trajectories.
    scale
        Factor used for scaling the arrows in the plot.
    figsize
        Width and height of the figures in inches.
    show_restriction_boundary
        Determines whether to also visualize the boundary of the domain restriction.
    """
    diffeomorphism = results['flow']
    dim = results['input'].dim

    if dim == 3:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, subplot_kw={'projection': '3d'})
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1, vals1 = results['input'].plot("Input", axis=ax1, figsize=figsize,
                                       show_restriction_boundary=show_restriction_boundary,
                                       restriction=results['restriction'])
    if not isinstance(vals1, list):
        fig.colorbar(vals1, ax=ax1, fraction=0.046, pad=0.04)
    ax2, vals2 = results['target'].plot("Target", axis=ax2, figsize=figsize,
                                        show_restriction_boundary=show_restriction_boundary,
                                        restriction=results['restriction'])
    if not isinstance(vals2, list):
        fig.colorbar(vals2, ax=ax2, fraction=0.046, pad=0.04)
    ax3, vals3 = results['transformed_input'].plot("Result", axis=ax3, figsize=figsize,
                                                   show_restriction_boundary=show_restriction_boundary,
                                                   restriction=results['restriction'])
    if not isinstance(vals3, list):
        fig.colorbar(vals3, ax=ax3, fraction=0.046, pad=0.04)
    diff = results['target'] - results['transformed_input']
    ax4, vals4 = diff.plot("Difference of target and result", axis=ax4, figsize=figsize,
                           show_restriction_boundary=show_restriction_boundary, restriction=results['restriction'])
    if not isinstance(vals4, list):
        fig.colorbar(vals4, ax=ax4, fraction=0.046, pad=0.04)
    plt.show()

    _ = results['vector_fields'].animate("Time-evolution of the vector field", interval=interval, scale=scale,
                                         figsize=figsize)
    plt.show()

    _, singular_values = pod(results['vector_fields'], return_singular_values='all')
    plt.semilogy(singular_values)
    plt.title("Singular values of time-evolution of the vector field")
    plt.show()

    results['initial_vector_field'].plot("Initial vector field", interval=interval, scale=scale, color_length=True,
                                         figsize=figsize)
    plt.show()

    results['initial_vector_field'].get_magnitude().plot("Magnitude of initial vector field", figsize=figsize)
    plt.show()

    if dim > 1:
        results['initial_vector_field'].get_angle().plot("Angle of initial vector field", figsize=figsize)
        plt.show()

    if dim == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': '3d'})
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1, vals1 = results['initial_vector_field'].plot("Initial vector field", axis=ax1, interval=interval,
                                                      scale=scale, color_length=True, figsize=figsize)
    if not isinstance(vals1, list):
        fig.colorbar(vals1, ax=ax1, fraction=0.046, pad=0.04)
    ax2, vals2 = results['initial_vector_field'].get_magnitude().plot("Magnitude of initial vector field",
                                                                      axis=ax2, figsize=figsize)
    if not isinstance(vals2, list):
        fig.colorbar(vals2, ax=ax2, fraction=0.046, pad=0.04)
    if dim > 1:
        ax3, vals3 = results['initial_vector_field'].get_angle().plot("Angle of initial vector field",
                                                                      axis=ax3, figsize=figsize)
        if not isinstance(vals3, list):
            fig.colorbar(vals3, ax=ax3, fraction=0.046, pad=0.04)
    plt.show()

    if results['initial_vector_field'].dim == 2:
        results['initial_vector_field'].plot_as_warpgrid("Initial vector field", interval=interval, figsize=figsize)
        plt.show()

        results['initial_vector_field'].plot_as_warpgrid("Initial vector field", interval=interval, figsize=figsize,
                                                         show_displacement_vectors=True)
        plt.show()

        diffeomorphism.plot("Diffeomorphism", interval=interval, figsize=figsize)
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        ax1, vals1 = results['initial_vector_field'].plot("Initial vector field", axis=ax1, interval=interval,
                                                          scale=scale, color_length=True, figsize=figsize)
        if not isinstance(vals1, list):
            fig.colorbar(vals1, ax=ax1, fraction=0.046, pad=0.04)
        ax2, vals2 = results['vector_fields'][-1].plot("Final vector field", axis=ax2, interval=interval, scale=scale,
                                                       color_length=True, figsize=figsize)
        if not isinstance(vals2, list):
            fig.colorbar(vals2, ax=ax2, fraction=0.046, pad=0.04)
        ax3, vals3 = (results['initial_vector_field'] - results['vector_fields'][-1]).plot("Difference", axis=ax3,
                                                                                           interval=interval,
                                                                                           scale=scale,
                                                                                           color_length=True,
                                                                                           figsize=figsize)
        if not isinstance(vals3, list):
            fig.colorbar(vals3, ax=ax3, fraction=0.046, pad=0.04)
        plt.show()

        time_dependent_diffeomorphism = results['vector_fields'].integrate(get_time_dependent_diffeomorphism=True)
        time_dependent_diffeomorphism.plot("Time-evolution of diffeomorphism", interval=interval, frequency=frequency,
                                           figsize=figsize)
        plt.show()

        assert time_dependent_diffeomorphism[-1] == diffeomorphism

        _ = time_dependent_diffeomorphism.animate("Animation of time-evolution of diffeomorphism", interval=interval,
                                                  figsize=figsize)
        plt.show()

        _ = time_dependent_diffeomorphism.animate_transformation(results['input'],
                                                                 "Animation of the transformation of the input",
                                                                 interval=interval, figsize=figsize,
                                                                 show_restriction_boundary=show_restriction_boundary,
                                                                 restriction=results['restriction'])
        plt.show()

        diffeomorphism.set_inverse(results['vector_fields'].integrate_backward())
        diffeomorphism.inverse.plot("Inverse diffeomorphism", interval=interval, figsize=figsize)
        plt.show()

        inverse_transformed_registration_result = results['transformed_input'].push_forward(diffeomorphism.inverse)
        inverse_transformed_registration_result.plot("Inverse transformed registration result", figsize=figsize,
                                                     show_restriction_boundary=show_restriction_boundary,
                                                     restriction=results['restriction'])
        plt.show()

        diff = results['input'] - inverse_transformed_registration_result
        diff.plot("Difference between input and inverse transformed registration result", figsize=figsize,
                  show_restriction_boundary=show_restriction_boundary, restriction=results['restriction'])
        plt.show()

        inverse_transformed_target = results['target'].push_forward(diffeomorphism.inverse)
        inverse_transformed_target.plot("Inverse transformed target", figsize=figsize,
                                        show_restriction_boundary=show_restriction_boundary,
                                        restriction=results['restriction'])
        plt.show()

        diff = results['input'] - inverse_transformed_target
        diff.plot("Difference between input and inverse transformed target", figsize=figsize,
                  show_restriction_boundary=show_restriction_boundary, restriction=results['restriction'])
        plt.show()


def save_plots_registration_results(results, filepath='results/', postfix='', interval=1, figsize=(20, 20),
                                    show_restriction_boundary=True, save_animations=False):
    """Saves some plots of the results from registration via geodesic shooting.

    Parameters
    ----------
    results
        Dictionary containing the results obtained via geodesic shooting.
    filepath
        Directory to save the images to.
    postfix
        String to add to the title of all plots.
    interval
        Interval in which to sample.
    figsize
        Width and height of the figures in inches.
    show_restriction_boundary
        Determines whether to also visualize the boundary of the domain restriction.
    save_animations
        Determines whether to also save animations.
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    results['input'].save(filepath + 'input.png', title='Input' + postfix, figsize=figsize,
                          show_restriction_boundary=show_restriction_boundary, restriction=results['restriction'])
    results['target'].save(filepath + 'target.png', title='Target' + postfix, figsize=figsize,
                           show_restriction_boundary=show_restriction_boundary, restriction=results['restriction'])
    results['transformed_input'].save(filepath + 'transformed_input.png', title='Result' + postfix, figsize=figsize,
                                      show_restriction_boundary=show_restriction_boundary,
                                      restriction=results['restriction'])
    diff = results['target'] - results['transformed_input']
    diff.save(filepath + 'difference.png', title='Difference of target and result' + postfix, figsize=figsize,
              show_restriction_boundary=show_restriction_boundary, restriction=results['restriction'])
    _, singular_values = pod(results['vector_fields'], return_singular_values='all')
    plt.semilogy(singular_values)
    plt.title('Singular values of time-evolution of the vector field')
    plt.savefig(filepath + 'singular_values_time_evolution_vector_field.png')
    plt.close()
    results['initial_vector_field'].save(filepath + 'initial_vector_field.png', plot_type='default',
                                         plot_args={'title': 'Initial vector field' + postfix, 'interval': interval,
                                                    'color_length': True, 'scale': None, 'figsize': figsize})
    results['initial_vector_field'].save_vtk(filepath + 'initial_vector_field_vtk')
    results['initial_vector_field'].save(filepath + 'initial_vector_field_streamlines.png', plot_type='streamlines',
                                         plot_args={'title': 'Initial vector field' + postfix, 'interval': interval,
                                                    'color_length': True, 'scale': None, 'figsize': figsize,
                                                    'density': 2})
    results['initial_vector_field'].get_magnitude().save(filepath + 'initial_vector_field_magnitude.png',
                                                         title='Magnitude of initial vector field' + postfix,
                                                         figsize=figsize)
    results['initial_vector_field'].get_angle().save(filepath + 'initial_vector_field_angle.png',
                                                     title='Angle of initial vector field' + postfix, figsize=figsize)
    for d in range(results['initial_vector_field'].dim):
        comp = results['initial_vector_field'].get_component_as_function(d)
        comp.save(filepath + f'initial_vector_field_component_{d}.png',
                  title='Initial vector field component ' + str(d) + postfix, figsize=figsize)
    diffeomorphism = results['flow']
    diffeomorphism.save(filepath + 'diffeomorphism.png', title='Diffeomorphism' + postfix,  figsize=figsize,
                        interval=interval)
    diffeomorphism.set_inverse(results['vector_fields'].integrate_backward())
    diffeomorphism.inverse.save(filepath + 'inverse_diffeomorphism.png', title='Inverse diffeomorphism' + postfix,
                                figsize=figsize, interval=interval)

    inverse_transformed_registration_result = results['transformed_input'].push_forward(diffeomorphism.inverse)
    inverse_transformed_registration_result.save(filepath + 'inverse_transformed_registration_result.png',
                                                 title='Inverse transformed registration result' + postfix,
                                                 figsize=figsize, show_restriction_boundary=show_restriction_boundary,
                                                 restriction=results['restriction'])
    diff = results['input'] - inverse_transformed_registration_result
    diff.save(filepath + 'diff_input_inverse_transformed_registration_result.png',
              title='Difference between input and inverse transformed registration result' + postfix,
              figsize=figsize, show_restriction_boundary=show_restriction_boundary, restriction=results['restriction'])
    inverse_transformed_target = results['target'].push_forward(diffeomorphism.inverse)
    inverse_transformed_target.save(filepath + 'inverse_transformed_target.png',
                                    title='Inverse transformed target' + postfix, figsize=figsize,
                                    show_restriction_boundary=show_restriction_boundary,
                                    restriction=results['restriction'])
    diff = results['input'] - inverse_transformed_target
    diff.save(filepath + 'diff_input_inverse_transformed_target.png',
              title='Difference between input and inverse transformed target' + postfix, figsize=figsize,
              show_restriction_boundary=show_restriction_boundary, restriction=results['restriction'])

    time_dependent_diffeomorphism = results['vector_fields'].integrate(get_time_dependent_diffeomorphism=True)
    assert time_dependent_diffeomorphism[-1] == diffeomorphism

    if save_animations:
        ani = time_dependent_diffeomorphism.animate('Animation of the time-evolution of the diffeomorphism' + postfix,
                                                    figsize=figsize, interval=interval)
        try:
            ani.save(filepath + 'animation_time_evolution_diffeomorphism.gif', writer='imagemagick',
                     fps=max(1, len(time_dependent_diffeomorphism) // 10))
        except Exception as e:
            print(f"Could not save animation! Error: {e}")
        plt.close()

        ani = time_dependent_diffeomorphism.animate_transformation(results['input'],
                                                                   'Animation of the transformation of the input'
                                                                   + postfix,
                                                                   figsize=figsize, interval=interval,
                                                                   show_restriction_boundary=show_restriction_boundary,
                                                                   restriction=results['restriction'])
        try:
            ani.save(filepath + 'animation_transformation_input.gif', writer='imagemagick',
                     fps=max(1, len(time_dependent_diffeomorphism) // 10))
        except Exception as e:
            print(f"Could not save animation! Error: {e}")
        plt.close()

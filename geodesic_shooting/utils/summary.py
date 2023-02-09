import os
import matplotlib.pyplot as plt


def plot_registration_results(results, interval=1, frequency=1, scale=None):
    """Plots some of the results from registration via geodesic shooting.

    Parameters
    ----------
    results
        Dictionary containing the results obtained via geodesic shooting.
    interval
        Interval in which to sample.
    """
    diffeomorphism = results['flow']

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1, vals1 = results['input'].plot("Input", axis=ax1)
    if not isinstance(vals1, list):
        fig.colorbar(vals1, ax=ax1, fraction=0.046, pad=0.04)
    ax2, vals2 = results['target'].plot("Target", axis=ax2)
    if not isinstance(vals2, list):
        fig.colorbar(vals2, ax=ax2, fraction=0.046, pad=0.04)
    ax3, vals3 = results['transformed_input'].plot("Result", axis=ax3)
    if not isinstance(vals3, list):
        fig.colorbar(vals3, ax=ax3, fraction=0.046, pad=0.04)
    ax4, vals4 = (results['target'] - results['transformed_input']).plot("Difference of target and result", axis=ax4)
    if not isinstance(vals4, list):
        fig.colorbar(vals4, ax=ax4, fraction=0.046, pad=0.04)
    plt.show()

    if results['vector_fields'].dim == 2:
        _ = results['vector_fields'].animate("Time-evolution of the vector field", interval=interval, scale=scale)
        plt.show()

        results['vector_fields'][0].plot("Initial vector field", interval=interval, scale=scale)
        plt.show()

        results['vector_fields'][0].plot_as_warpgrid("Initial vector field", interval=interval)
        plt.show()

        results['vector_fields'][0].plot_as_warpgrid("Initial vector field", interval=interval,
                                                     show_displacement_vectors=True)
        plt.show()

        diffeomorphism.plot("Diffeomorphism", interval=interval)
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1 = results['vector_fields'][0].plot("Initial vector field", axis=ax1, interval=interval, scale=scale)
        ax2 = results['vector_fields'][-1].plot("Final vector field", axis=ax2, interval=interval, scale=scale)
        ax3 = (results['vector_fields'][0] - results['vector_fields'][-1]).plot("Difference", axis=ax3,
                                                                                interval=interval, scale=scale)
        plt.show()

        time_dependent_diffeomorphism = results['vector_fields'].integrate(get_time_dependent_diffeomorphism=True)
        time_dependent_diffeomorphism.plot("Time-evolution of diffeomorphism", interval=interval, frequency=frequency)
        plt.show()

        assert time_dependent_diffeomorphism[-1] == diffeomorphism

        _ = time_dependent_diffeomorphism.animate("Animation of time-evolution of diffeomorphism", interval=interval)
        plt.show()

        diffeomorphism.set_inverse(results['vector_fields'].integrate_backward())
        diffeomorphism.inverse.plot("Inverse diffeomorphism", interval=interval)
        plt.show()

        inverse_transformed_registration_result = results['transformed_input'].push_forward(diffeomorphism.inverse)
        inverse_transformed_registration_result.plot("Inverse transformed registration result")
        plt.show()

        inverse_transformed_target = results['target'].push_forward(diffeomorphism.inverse)
        inverse_transformed_target.plot("Inverse transformed target")
        plt.show()


def save_plots_registration_results(results, filepath='results/'):
    """Saves some plots of the results from registration via geodesic shooting.

    Parameters
    ----------
    results
        Dictionary containing the results obtained via geodesic shooting.
    filepath
        Directory to save the images to.
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    results['input'].save(filepath + 'input.png', title="Input")
    results['target'].save(filepath + 'target.png', title="Target")
    results['transformed_input'].save(filepath + 'transformed_input.png', title="Result")
    results['initial_vector_field'].save(filepath + 'initial_vector_field.png', plot_type='default',
                                         plot_args={'title': "Initial vector field", 'color_length': True,
                                                    'show_axis': True})
    results['initial_vector_field'].save(filepath + 'initial_vector_field_streamlines.png', plot_type='streamlines',
                                         plot_args={'title': "Initial vector field", 'color_length': True,
                                                    'show_axis': True, 'density': 2})
    results['flow'].save(filepath + 'diffeomorphism.png', title="Diffeomorphism")
    inverse_diffeomorphism = results['vector_fields'].integrate_backward()
    inverse_diffeomorphism.save(filepath + 'inverse_diffeomorphism.png', title="Inverse diffeomorphism")

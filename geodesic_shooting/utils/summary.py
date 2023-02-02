import os
import matplotlib.pyplot as plt


def plot_registration_results(results, interval=1, scale=None):
    """Plots some of the results from registration via geodesic shooting.

    Parameters
    ----------
    results
        Dictionary containing the results obtained via geodesic shooting.
    interval
        Interval in which to sample.
    """
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

        fig, ax = plt.subplots(1, 1)
        ax = results['vector_fields'][0].plot("Initial vector field", axis=ax, scale=scale)
        plt.show()

        _ = results['flow'].plot_as_warpgrid("Flow")
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1 = results['vector_fields'][0].plot("Initial vector field", axis=ax1, scale=scale)
        ax2 = results['vector_fields'][-1].plot("Final vector field", axis=ax2, scale=scale)
        ax3 = (results['vector_fields'][0] - results['vector_fields'][-1]).plot("Difference", axis=ax3, scale=scale)
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
    results['initial_vector_field'].save(filepath + 'initial_vector_field.png', title="Initial vector field")
    results['flow'].save(filepath + 'flow.png', title="Flow")

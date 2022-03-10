import os
import matplotlib.pyplot as plt


def plot_registration_results(results, interval=1):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    results['input'].plot("Input", axis=ax1)
    results['target'].plot("Target", axis=ax2)
    results['transformed_input'].plot("Result", axis=ax3)
    (results['target'] - results['transformed_input']).plot("Difference of target and result", axis=ax4)
    plt.show()

    _ = results['vector_fields'].animate()
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1 = results['vector_fields'][0].plot("Initial vector field", axis=ax1)
    ax2 = results['vector_fields'][-1].plot("Final vector field", axis=ax2)
    ax3 = (results['vector_fields'][0] - results['vector_fields'][-1]).plot("Difference", axis=ax3)
    plt.show()

    results['flow'].plot_as_warpgrid(title="Inverse warp grid", interval=interval)
    plt.show()


def save_plots_registration_results(results, filepath='results/'):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    results['input'].save(filepath + 'input.png', title="Input")
    results['target'].save(filepath + 'target.png', title="Target")
    results['transformed_input'].save(filepath + 'transformed_input.png', title="Result")
    results['initial_vector_field'].save(filepath + 'initial_vector_field.png', title="Initial vector field")

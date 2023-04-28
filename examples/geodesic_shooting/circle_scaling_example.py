import numpy as np

import geodesic_shooting
from geodesic_shooting.core import VectorField, Diffeomorphism, TimeDependentScalarFunction
from geodesic_shooting.utils.create_example_images import make_circle
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


if __name__ == "__main__":
    # create images
    template = make_circle((64, 64), np.array([32, 32]), 10)
    target = make_circle((64, 64), np.array([32, 32]), 20)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.05, exponent=1, gamma=5.)
    result = gs.register(template, target, sigma=0.01, return_all=True,
                         optimizer_options={'disp': True, 'maxiter': 20})

    plot_registration_results(result)
    cn_initial = gs.regularizer.cauchy_navier(result['initial_vector_field'])
    cn_initial.plot(title='Cauchy Navier operator applied to initial vector field', color_length=True)
    cn_initial.get_magnitude().plot(title='Magnitude of Cauchy Navier operator applied to initial vector field')

    magnitude_evo_cn_vector_fields = []
    evolution_transformed_template = []
    for vf, diffeo in zip(result['vector_fields'].to_numpy(), result['vector_fields'].integrate(get_time_dependent_diffeomorphism=True).to_numpy()):
        magnitude_evo_cn_vector_fields.append(gs.regularizer.cauchy_navier(VectorField(data=vf)).get_magnitude())
        evolution_transformed_template.append(template.push_forward(Diffeomorphism(data=diffeo)))

    magnitude_evo_cn_vector_fields = TimeDependentScalarFunction(data=magnitude_evo_cn_vector_fields)
    ani1 = magnitude_evo_cn_vector_fields.animate(title='Evolution of magnitude of Cauchy Navier operator applied to vector fields')
    evolution_transformed_template = TimeDependentScalarFunction(data=evolution_transformed_template)
    ani2 = evolution_transformed_template.animate(title='Evolution of transformed template')
    import matplotlib.pyplot as plt
    plt.show()

    save_plots_registration_results(result, filepath='results_circle_scaling/')

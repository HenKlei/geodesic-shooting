import geodesic_shooting
from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils.summary import plot_registration_results


if __name__ == "__main__":
    # define greyscale images
    N = 10
    M = 5
    template = ScalarFunction((N, M))
    target = ScalarFunction((N, M))
    template[N//5:2*N//5, M//5:2*M//5] = 1
    target[2*N//5:3*N//5, M//5:2*M//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=0.1, exponent=1)
    result = gs.register(template, target, sigma=0.01, return_all=True)

    plot_registration_results(result)

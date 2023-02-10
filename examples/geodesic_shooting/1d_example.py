import geodesic_shooting
from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils.summary import plot_registration_results


if __name__ == "__main__":
    # define greyscale images
    N = 100
    input_ = ScalarFunction((N,))
    target = ScalarFunction((N,))
    input_[N//5:2*N//5] = 1
    target[2*N//5:3*N//5] = 1

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=4., exponent=2)
    result = gs.register(input_, target, sigma=0.01, return_all=True)

    plot_registration_results(result)

from geodesic_shooting.utils import sampler


def push_forward(image, flow):
    """Pushes forward an image along a flow.

    Parameters
    ----------
    image
        `ScalarFunction` to push forward.
    flow
        `VectorField` containing the flow according to which to push the input forward.

    Returns
    -------
    `ScalarFunction` of the forward-pushed image.
    """
    return sampler.sample(image, flow)

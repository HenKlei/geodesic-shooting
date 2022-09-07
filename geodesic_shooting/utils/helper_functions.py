def tuple_product(val):
    """Product of the entries of a tuple.

    This is particularly helpful for computing the number of entries in
    an array when given the shape of the array.

    Parameters
    ----------
    val
        Tuple of whose entries the product is supposed to be computed.

    Returns
    -------
    Computed product of the entries in the tuple.
    """
    res = 1
    for ele in val:
        res *= ele
    return res


def lincomb(modes, coefficients):
    """Linear combination of vectors given associated coefficients.

    Parameters
    ----------
    modes
        The vectors to linearly combine.
    coefficients
        The coefficients to use for the linear combination.

    Returns
    -------
    The linear combination of the modes given the coefficients.
    """
    assert len(modes) == len(coefficients)
    assert len(modes) > 0
    type_input = type(modes[0])

    res = type_input(spatial_shape=modes[0].spatial_shape)
    for v, c in zip(modes, coefficients):
        res += c * v
    return res

def tuple_product(val):
    res = 1
    for ele in val:
        res *= ele
    return res


def lincomb(modes, coefficients):
    assert len(modes) == len(coefficients)
    assert len(modes) > 0
    type_input = type(modes[0])

    res = type_input(spatial_shape=modes[0].spatial_shape)
    for v, c in zip(modes, coefficients):
        res += c * v
    return res

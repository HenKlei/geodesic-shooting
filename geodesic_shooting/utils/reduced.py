import numpy as np


def lincomb(rb_velocity_fields, coefficients):
    assert coefficients.ndim == 1
    assert rb_velocity_fields.shape[0] == len(coefficients)

    return np.tensordot(rb_velocity_fields, coefficients, axes=([0], [0]))

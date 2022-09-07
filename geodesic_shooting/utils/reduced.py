import numpy as np

from geodesic_shooting.core import VectorField


def lincomb(rb_vector_fields, coefficients):
    assert coefficients.ndim == 1
    assert len(rb_vector_fields) == len(coefficients)
    assert len(rb_vector_fields) > 0

    res = VectorField(rb_vector_fields[0].spatial_shape)
    for v, c in zip(rb_vector_fields, coefficients):
        res += c * v
    return res

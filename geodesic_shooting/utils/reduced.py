import numpy as np

from geodesic_shooting.core import VectorField


def lincomb(rb_velocity_fields, coefficients):
    assert coefficients.ndim == 1
    assert len(rb_velocity_fields) == len(coefficients)

    res = np.zeros(rb_velocity_fields[0].shape)
    for v, c in zip(rb_velocity_fields, coefficients):
        res += c * v.to_numpy()
    return VectorField(data=res)

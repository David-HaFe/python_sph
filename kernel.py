

import numpy as np
from diagnostics import diagnostics

# SPH smoothing length
h = .7
kernel_radius = h
sigma_W = .01
h_dim = 1

# kernel for a given point and a reference point
# using C² Wendland kernel
def W(x_a: np.array, x_b: np.array):
    diagnostics.time_W()

    distance = np.linalg.norm(x_a - x_b)/h

    if distance < kernel_radius:
        result = (sigma_W/h_dim) * (
            .125*distance**5
            - .9375*distance**4
            + 2.5*distance**3
            - 2.5*distance**2
            + 1
        )
    else:
        result = 0

    # if distance small enough, calculate kernel, else return 0
    # result = jnp.where(
    #     distance < kernel_radius,
    #     sigma_W/h_dim * (1 + 2*distance)*(1 - .5*distance)**4,
    #     0.0,
    # )
    diagnostics.time_W()
    return result

# gradient of the kernel function
def delta_W(x_a: np.array, x_b: np.array):
    diagnostics.time_delta_W()

    distance = np.linalg.norm(x_a - x_b)/h

    if distance < kernel_radius:
        difference = x_a - x_b
        result = (sigma_W/h_dim) * (
            (5/h**2)*difference
            - (7.5/h**3)*difference**2
            + (3.75/h**4)*difference**3
            - (.625/h**5)*difference**4
        )
    else:
        result = np.array([0, 0])

    diagnostics.time_delta_W()
    return + result



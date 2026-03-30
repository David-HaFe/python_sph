

import numpy as np
from diagnostics import diagnostics
from model_parameters import model_parameters

# SPH smoothing length
h = 1
kernel_radius = 2*h
sigma_W = .01
h_dim = h^2
kernel_zero_tolerance = 1e-5

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
def gradient_W(x_a: np.array, x_b: np.array):
    diagnostics.time_gradient_W()

    distance = np.linalg.norm(x_a - x_b)/h

    if distance < kernel_zero_tolerance:
        result = np.zeros(2)

    elif distance < kernel_radius:
        difference = x_a - x_b
        result = (sigma_W/h_dim) * (
            (5/h**2)*difference
            - (7.5/h**3)*difference**2
            + (3.75/h**4)*difference**3
            - (.625/h**5)*difference**4
        )

    else:
        result = np.zeros(2)

    diagnostics.time_gradient_W()
    return (+result)

# normalised gradient of the kernel function
# args:
#   x_a: kernel center position in R²
#   x_b: point at which gradient is evaluated in R²
#   x:   state vector for L matrix in R^n, n = no particles
def normalised_gradient_W(x_a: np.array, x_b: np.array, x: np.array):
    diagnostics.time_norm_gradient_W()
    m = model_parameters.m
    rho = model_parameters.rho

    L_inv = np.zeros((2,2))
    gradient_W_ab = gradient_W(x_a, x_b)

    for i, (x_i) in enumerate(zip(x)):
        gradient = (gradient_W(x_a, x_i)).reshape(-1, 2)
        L_inv += np.matmul(np.transpose((m[i]/rho[i])*(x_i - x_a)), gradient)

    gradient_W_ab = gradient_W_ab.reshape(-1, 1)

    L = np.linalg.inv(L_inv)
    result = np.matmul(L, gradient_W_ab)

    diagnostics.time_norm_gradient_W()
    return result



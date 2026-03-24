

import jax.numpy as jnp
import jax

from diagnostics import diagnostics

# SPH smoothing length
h = 1

# kernel for a given point and a reference point
# using C² Wendland kernel
def W(x_a: jnp.array, x_b: jnp.array):
    diagnostics.time_W()
    kernel_radius = 2
    sigma_W = 1
    h_dim = 1

    distance = jnp.linalg.norm(x_a - x_b + 1e-8)/h

    # if distance small enough, calculate kernel, else return 0
    result = jnp.where(
        distance < kernel_radius,
        sigma_W/h_dim * (1 + 2*distance)*(1 - .5*distance)**4,
        0.0,
    )
    diagnostics.time_W()
    return result

# gradient of the kernel function
def delta_W(x_a: jnp.array, x_b: jnp.array):
    diagnostics.time_delta_W()
    result = jax.grad(W, argnums=1)(x_a, x_b)
    diagnostics.time_delta_W()
    return result



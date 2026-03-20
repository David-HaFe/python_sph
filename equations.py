

# equations needed for the sph implementation

# import numpy as np
import numpy as np
import jax.numpy as jnp
import jax

from diagnostics import diagnostics

h = 1   # -> SPH smoothing length

# implements the discretized navier stokes equation for the ode solver
def navier_stokes(q, t):
    diagnostics.time_navier_stokes()
    no_particles = np.size(q)//2
    q_dot = np.array([])

    # TODO: these have to be updated, not sure how to do that
    m = np.full(no_particles, 1.0)
    rho = np.full(no_particles, 2.0)
    p = np.full(no_particles, 1.0)
    mu = np.full(no_particles, 1.0)

    for a in range(0, no_particles):
        # reset everything and get new vectors
        pressure_term = 0
        viscosity_term = 0
        x_a = q[2*a]
        v_a = q[2*a+1]

        for b in range(0, no_particles):
            x_b = q[2*b]
            particle_close_enough = W(x_a, x_b) > 0

            if not (a == b):
                diagnostics.register_particle(particle_close_enough)

            if (particle_close_enough and not (a == b)):
                v_b = q[2*b+1]

                pressure_term += m[b] * (
                    p[b]/jnp.power(rho[b], 2)
                    + p[a]/jnp.power(rho[a], 2)
                    ) * delta_W(x_a, x_b)

                delta_x = x_a - x_b
                delta_v = v_a - v_b

                viscosity_term += m[b] * (
                    (mu[a] + mu[b])/(rho[a]*rho[b]) * jnp.dot(delta_x, delta_v)
                    / (jnp.dot(delta_x, delta_x) + .01*h**2)
                    ) * delta_W(x_a, x_b)

        # fill solution array
        q_dot = np.append(q_dot, v_a)
        q_dot = np.append(q_dot, rho[a]*(pressure_term + viscosity_term))

    diagnostics.time_navier_stokes()
    return q_dot

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
        sigma_W/h_dim * (1 + 2*distance)*jnp.power((1 - .5*distance), 4),
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



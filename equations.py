

# equations needed for the sph implementation

# import numpy as np
import numpy as np
import jax.numpy as jnp

no_particles = 4

# implements the discretized navier stokes equation for the ode solver
def navier_stokes(q, t, m, rho, p, mu):
    q_dot = np.array([])

    for a in range(0, no_particles):
        # reset everything and get new vectors
        pressure_term = 0
        viscosity_term = 0
        x_a = q[2*a]
        v_a = q[2*a+1]

        for b in range(0, no_particles):
            x_b = q[2*b]
            if (W(x_a, x_b) > 0 and not(a == b)):
                v_b = q[2*b+1]

                pressure_term += m[b] * (
                    p[b]/jnp.power(rho[b], 2)
                    + p[a]/jnp.power(rho[a], 2)
                    ) * delta_W(x_a, x_b)

                delta_x = x_a - x_b
                delta_v = v_a - v_b

                viscosity_term += m[b] * (
                    (mu[a] + mu[b])/(rho[a]*rho[b])
                    * (delta_x*delta_v)/(jnp.power(delta_x, 2) + .01*h**2)
                    ) * delta_W(x_a, x_b)

        # fill solution array
        np.append(q_dot, v_a)
        np.append(q_dot, rho[a]*(pressure_term + viscosity_term))

    return q_dot

# kernel for a given point and a reference point
# using C² Wendland kernel
def W(x_a: jnp.array, x_b: jnp.array):
    h = 1   # -> SPH smoothing length
    kernel_radius = 2
    sigma_W = 1
    h_dim = 1

    distance = jnp.linalg.norm(x_a - x_b)/h

    return jnp.where(distance < kernel_radius,
        sigma_W/h_dim * (1 + 2*distance)*jnp.power((1 - .5*distance), 4),
        0.0)

# gradient of the kernel function
def delta_W(x_a: jnp.array, x_b: jnp.array):
    return jax.grad(W, argnums=1)(x_a, x_b)



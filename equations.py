

# equations needed for the sph implementation

# import numpy as np
import numpy as np
import jax.numpy as jnp
import sys

from diagnostics import diagnostics
from kernel import W, delta_W, h

gravity = np.array([0, -9.81])

# implements the discretized navier stokes equation for the ode solver
# y consists of the triple [positions, velocities ,density] for each particle,
# all concatenated like this one after the other
def navier_stokes(t, y):
    diagnostics.time_navier_stokes()
    no_particles = np.size(y)//(2+2+1)
    y_dot = np.zeros(np.size(y))
    x_dot = np.zeros((no_particles, 2))
    v_dot = np.zeros((no_particles, 2))
    rho_dot = np.zeros(no_particles)

    # TODO: these have to be updated, not sure how to do that
    m = np.full(no_particles, 1.0)
    mu = np.full(no_particles, 1.0)

    x = y[: 2*no_particles]
    v = y[2*no_particles : 4*no_particles]
    rho = y[4*no_particles :]
    x = x.reshape(-1, 2)
    v = v.reshape(-1, 2)

    for a, (x_a, v_a, rho_a) in enumerate(zip(x, v, rho)):
        # reset everything and get new vectors
        x_a = jnp.array(x_a)
        v_a = jnp.array(v_a)
        pressure_term = jnp.zeros(2)
        viscosity_term = jnp.zeros(2)
        rho_dot_a = 0

        p_a = calculate_pressure(rho_a)

        for b, (x_b, v_b, rho_b) in enumerate(zip(x, v, rho)):
            x_b = jnp.array(x_b)
            particle_close_enough = W(x_a, x_b) > 0

            if not (a == b):
                diagnostics.register_particle(particle_close_enough)

            # if particle_close_enough:
            if (particle_close_enough and not (a == b)):

                # calculate dv/dt
                v_b = jnp.array(v_b)
                p_b = calculate_pressure(rho_b)

                pressure_term += m[b] * (
                    p_b/jnp.power(rho_b, 2)
                    + p_a/jnp.power(rho_a, 2)
                    ) * delta_W(x_a, x_b)

                delta_x = x_a - x_b
                delta_v = v_a - v_b

                viscosity_term += m[b] * (
                    (mu[a] + mu[b])/(rho_a*rho_b) * jnp.dot(delta_x, delta_v)
                    / (jnp.dot(delta_x, delta_x) + .01*h**2)
                    ) * delta_W(x_a, x_b)

                # calculate d(rho)/dt
                rho_dot_a += m[b]/rho_b*jnp.dot(v_b - v_a, delta_W(x_a, x_b))

        # fill solution array
        x_dot[a] = v_a
        v_dot[a] = - pressure_term + viscosity_term
        rho_dot[a] = - rho_a * rho_dot_a

    x_dot = x_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    y_dot = np.concatenate((x_dot, v_dot, rho_dot))

    # sys.stdout.write(f"\r\033[K{t}")
    # sys.stdout.flush()

    diagnostics.time_navier_stokes()
    return y_dot

# calculates the pressure based on given formula
def calculate_pressure(rho):
    gamma = 7
    c_0 = 10
    # TODO: maybe refine reference density, depending on what it turns
    #       out to be
    rho_0 = 1.1
    # TODO: find out what p_B actually is
    p_B = 0

    return (((c_0**2)*rho_0/gamma) * (
        ((rho/rho_0)**gamma)-1) + p_B)



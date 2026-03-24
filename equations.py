

# equations needed for the sph implementation

# import numpy as np
import numpy as np
import jax.numpy as jnp

from diagnostics import diagnostics
from kernel import W, delta_W, h

gravity = np.array([0, -9.81])

# implements the discretized navier stokes equation for the ode solver
# q consists of the triple [position,velocity,density] for each particle, all concatenated like this one after the other
def navier_stokes(t, q, is_wall_particle):
    diagnostics.time_navier_stokes()
    no_particles = np.size(q)//5
    q_dot = np.array([])

    # TODO: these have to be updated, not sure how to do that
    m = np.full(no_particles, 1.0)
    mu = np.full(no_particles, 1.0)

    for a in range(0, no_particles):
        if not is_wall_particle[a]:
            # reset everything and get new vectors
            pressure_term = 0
            viscosity_term = 0
            rho_dot = 0

            x_a = jnp.array([q[5*a], q[5*a+1]])
            v_a = jnp.array([q[5*a+2], q[5*a+3]])
            rho_a = q[5*a+4]
            p_a = calculate_pressure(rho_a)

            for b in range(0, no_particles):
                x_b = jnp.array([q[5*b], q[5*b+1]])
                particle_close_enough = W(x_a, x_b) > 0

                if not (a == b):
                    diagnostics.register_particle(particle_close_enough)

                # if (particle_close_enough and not (a == b)):
                if particle_close_enough:

                    # calculate dv/dt
                    v_b = jnp.array([q[5*b+2], q[5*b+3]])
                    rho_b = q[5*b+4]
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

                    # calculate d\rho/dt
                    rho_dot += m[b]/rho_b*jnp.dot(v_b - v_a, delta_W(x_a, x_b))

            # fill solution array
            q_dot = np.append(q_dot, v_a)
            q_dot = np.append(q_dot, - pressure_term + viscosity_term)
            # q_dot = np.append(q_dot, g - pressure_term + viscosity_term)
            q_dot = np.append(q_dot, rho_dot)

        else:
            # is wall particle, just apply dummy solution
            q_dot = np.append(q_dot, np.array([0, 0, 0, 0, 100]))

    diagnostics.time_navier_stokes()
    return q_dot

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



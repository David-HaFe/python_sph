

# equations needed for the sph implementation

# import numpy as np
import numpy as np
import jax.numpy as jnp

from diagnostics import diagnostics
from kernel import W, delta_W, h

gravity = np.array([0, -9.81])

# implements the discretized navier stokes equation for the ode solver
# q consists of the triple [position,velocity,density] for each particle, all concatenated like this one after the other
# def navier_stokes(t, q, is_wall_particle=None):
def navier_stokes(t, y, is_wall_particle=None):
    diagnostics.time_navier_stokes()
    no_particles = np.size(y)//(2 + 2 + 1)
    # y_dot = []
    # x_dot = np.zeros(2 * no_particles)
    # u_dot = np.zeros(2 * no_particles)

    x = y[:2 * no_particles]
    u = y[2 * no_particles : 4 * no_particles]
    rho = y[4 * no_particles :]
    X = x.reshape(-1, 2)
    U = u.reshape(-1, 2)

    X_dot = np.zeros((no_particles, 2))
    U_dot = np.zeros((no_particles, 2))
    rho_dot = np.zeros(no_particles)

    # TODO: these have to be updated, not sure how to do that
    m = np.full(no_particles, 1.0)
    mu = np.full(no_particles, 1.0)

    diagnostics.time_navier_stokes()
    for i, (Xi, Ui, rhoi) in enumerate(zip(X, U, rho)):
        pressure_term = jnp.zeros(2)
        viscosity_term = jnp.zeros(2)
        rho_doti = 0

        Xi = jnp.array(Xi)
        pi = calculate_pressure(rhoi)
        for j, (Xj, Uj, rhoj) in enumerate(zip(X, U, rho)):
            if i == j:
                continue
            Xj = jnp.array(Xj)
            
            particle_close_enough = W(Xi, Xj) > 0
            diagnostics.register_particle(particle_close_enough)

            if particle_close_enough:
                Uj = jnp.array(Uj)
                pj = calculate_pressure(rhoj)

                pressure_term += m[j] * (
                    pj / jnp.power(rhoj, 2) + pi / jnp.power(rhoi, 2)
                ) * delta_W(Xi, Xj)

                delta_X = Xi - Xj
                delta_U = Ui - Uj

                viscosity_term += m[j] * (
                    (mu[i] + mu[j]) / (rhoi * rhoj) * jnp.dot(delta_X, delta_U)
                    / (jnp.dot(delta_X, delta_X) + .01 * h**2)
                ) * delta_W(Xi, Xj)

                rho_doti += m[j] / rhoj * jnp.dot(Uj - Ui, delta_W(Xi, Xj))

        X_dot[i] = Ui
        U_dot[i] = -pressure_term + viscosity_term
        rho_dot[i] = rho_doti
    
    diagnostics.time_navier_stokes()        

    x_dot = X_dot.reshape(-1, order="C")
    u_dot = U_dot.reshape(-1, order="C")
    y_dot = np.concatenate((x_dot, u_dot, rho_dot))
    return y_dot

    # for a in range(no_particles):
    #     # if not is_wall_particle[a]:
    #     if True:
    #         # reset everything and get new vectors
    #         pressure_term = 0
    #         viscosity_term = 0
    #         rho_dot = 0

    #         x_a = jnp.array([q[5*a], q[5*a+1]])
    #         v_a = jnp.array([q[5*a+2], q[5*a+3]])
    #         rho_a = q[5*a+4]
    #         p_a = calculate_pressure(rho_a)

    #         for b in range(0, no_particles):
    #             x_b = jnp.array([q[5*b], q[5*b+1]])
    #             particle_close_enough = W(x_a, x_b) > 0

    #             if not (a == b):
    #                 diagnostics.register_particle(particle_close_enough)

    #             # if (particle_close_enough and not (a == b)):
    #             if particle_close_enough:

    #                 # calculate dv/dt
    #                 v_b = jnp.array([q[5*b+2], q[5*b+3]])
    #                 rho_b = q[5*b+4]
    #                 p_b = calculate_pressure(rho_b)

    #                 pressure_term += m[b] * (
    #                     p_b/jnp.power(rho_b, 2)
    #                     + p_a/jnp.power(rho_a, 2)
    #                     ) * delta_W(x_a, x_b)

    #                 delta_X = x_a - x_b
    #                 delta_U = v_a - v_b

    #                 viscosity_term += m[b] * (
    #                     (mu[a] + mu[b])/(rho_a*rho_b) * jnp.dot(delta_X, delta_U)
    #                     / (jnp.dot(delta_X, delta_X) + .01*h**2)
    #                     ) * delta_W(x_a, x_b)

    #                 # calculate d\rho/dt
    #                 rho_dot += m[b]/rho_b*jnp.dot(v_b - v_a, delta_W(x_a, x_b))

    #         # fill solution array
    #         y_dot = np.append(y_dot, v_a)
    #         y_dot = np.append(y_dot, - pressure_term + viscosity_term)
    #         # q_dot = np.append(q_dot, g - pressure_term + viscosity_term)
    #         y_dot = np.append(y_dot, rho_dot)

    #     else:
    #         # is wall particle, just apply dummy solution
    #         y_dot = np.append(y_dot, np.array([0, 0, 0, 0, 100]))

    # diagnostics.time_navier_stokes()
    # return y_dot

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



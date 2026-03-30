

# equations needed for the sph implementation

# import numpy as np
import numpy as np
import sys

from diagnostics import diagnostics
from kernel import W, gradient_W, normalised_gradient_W
from model_parameters import model_parameters

# implements the discretized navier stokes equation for the ode solver
# y consists of the triple [positions, velocities, density] for each particle,
# all concatenated like this one after the other
# this is the INCOMPRESSIBLE case
def navier_stokes_incompressible(t, y, is_wall_particle):
    diagnostics.time_navier_stokes()
    no_particles = np.size(y)//(2+2)
    y_dot = np.zeros(np.size(y))
    x_dot = np.zeros((no_particles, 2))
    v_dot = np.zeros((no_particles, 2))

    x = y[: 2*no_particles]
    v = y[2*no_particles :]
    x = x.reshape(-1, 2)
    v = v.reshape(-1, 2)

    m = model_parameters.m
    nu = model_parameters.nu
    rho = model_parameters.rho

    for a, (x_a, v_a) in enumerate(zip(x, v)):
        if not is_wall_particle[a]:
            # reset everything and get new vectors
            viscosity_term = np.zeros(2)

            for b, (x_b, v_b) in enumerate(zip(x, v)):
                particle_close_enough = W(x_a, x_b) > 0

                if not (a == b):
                    diagnostics.register_particle(particle_close_enough)

                # particle is close enough
                if (particle_close_enough and not (a == b)):

                    # calculate dv/dt
                    delta_x = x_a - x_b
                    delta_v = v_a - v_b

                    # TODO: this actually uses the preliminary position, which
                    #       does not show up in every paper. I will use the
                    #       initial  position, but I should keep in mind that
                    #       this is not exactly as in Chow 2018.
                    viscosity_term += m[b]/rho[b] * (
                        (2*nu[b] + np.dot(delta_x, delta_v))
                        /(np.dot(delta_x, delta_x) + model_parameters.eta**2)
                    )

            # fill solution array
            x_dot[a] = v_a
            v_dot[a] = model_parameters.gravity + (
                viscosity_term * gradient_W(x_a, x_b)
            )
        else:
            x_dot[a] = np.zeros(2)
            v_dot[a] = np.zeros(2)

    x_dot = x_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    y_dot = np.concatenate((x_dot, v_dot))

    sys.stdout.write(f"\r\033[K{t}")
    sys.stdout.flush()

    diagnostics.time_navier_stokes()
    return y_dot

# evaluate the poisson pressure equations for given state
def poisson_pressure_equation(t, y, is_wall_particle, dt):
    diagnostics.time_poisson()
    no_particles = np.size(y)//(2+2)
    delta_p = np.empty(no_particles)

    x_dot = np.zeros((no_particles, 2))
    v_dot = np.zeros((no_particles, 2))

    x = y[: 2*no_particles]
    x = x.reshape(-1, 2)
    v = y[2*no_particles :]
    v = v.reshape(-1, 2)

    m = model_parameters.m
    nu = model_parameters.nu
    rho = model_parameters.rho

    for a, (x_a, v_a) in enumerate(zip(x, v)):
        if not is_wall_particle[a]:
            divergence_v = 0

            for b, (x_b, v_b) in enumerate(zip(x, v)):
                particle_close_enough = W(x_a, x_b) > 0

                # particle is close enough
                if (particle_close_enough and not (a == b)):
                    divergence_v += (m[b]/rho[b])*(
                        (v_b - v_a)
                    )

            divergence_v = divergence_v.reshape(-1, 2)
            divergence_v = np.dot(
                divergence_v, normalised_gradient_W(x_a, x_b, x)
            )
            v_dot[a] -= dt*divergence_v
        else:
            v_dot[a] = np.zeros(2)

    x_dot = x_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    y_dot = np.concatenate((x_dot, v_dot))

    diagnostics.time_poisson()
    return y_dot

# implements the discretized navier stokes equation for the ode solver
# y consists of the triple [positions, velocities, density] for each particle,
# all concatenated like this one after the other
# this is the COMPRESSIBLE case
def navier_stokes_compressible(t, y, is_wall_particle):
    diagnostics.time_navier_stokes()
    no_particles = np.size(y)//(2+2+1)
    y_dot = np.zeros(np.size(y))
    x_dot = np.zeros((no_particles, 2))
    v_dot = np.zeros((no_particles, 2))
    rho_dot = np.zeros(no_particles)

    # TODO: these have to be updated, not sure how to do that
    m = model_parameters.m
    mu = np.full(no_particles, .1)

    x = y[: 2*no_particles]
    v = y[2*no_particles : 4*no_particles]
    rho = y[4*no_particles :]
    x = x.reshape(-1, 2)
    v = v.reshape(-1, 2)

    for a, (x_a, v_a, rho_a) in enumerate(zip(x, v, rho)):
        if not is_wall_particle[a]:
            # reset everything and get new vectors
            pressure_term = np.zeros(2)
            viscosity_term = np.zeros(2)
            rho_dot_a = 0

            p_a = calculate_pressure(rho_a)

            for b, (x_b, v_b, rho_b) in enumerate(zip(x, v, rho)):
                # x_b = np.array(x_b)
                particle_close_enough = W(x_a, x_b) > 0

                if not (a == b):
                    diagnostics.register_particle(particle_close_enough)

                # if particle_close_enough:
                if (particle_close_enough and not (a == b)):

                    # calculate dv/dt
                    # v_b = np.array(v_b)
                    p_b = calculate_pressure(rho_b)

                    pressure_term += m[b] * (
                        p_b/np.power(rho_b, 2)
                        + p_a/np.power(rho_a, 2)
                        ) * gradient_W(x_a, x_b)

                    delta_x = x_a - x_b
                    delta_v = v_a - v_b

                    viscosity_term += m[b] * (
                        (mu[a] + mu[b])/(rho_a*rho_b)*np.dot(delta_x, delta_v)
                        / (np.dot(delta_x, delta_x) + .01*eta**2)
                        ) * gradient_W(x_a, x_b)

                    # calculate d(rho)/dt
                    rho_dot_a += (m[b]/rho_b)*(
                        np.dot(v_b - v_a, gradient_W(x_a, x_b))
                    )

            # fill solution array
            x_dot[a] = v_a
            v_dot[a] = gravity - pressure_term + viscosity_term
            rho_dot[a] = - rho_a * rho_dot_a
        else:
            x_dot[a] = np.zeros(2)
            v_dot[a] = np.zeros(2)
            rho_dot[a] = 0

    x_dot = x_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    y_dot = np.concatenate((x_dot, v_dot, rho_dot))

    sys.stdout.write(f"\r\033[K{t}")
    sys.stdout.flush()

    diagnostics.time_navier_stokes()
    return y_dot

# calculates the pressure based on given formula
def calculate_pressure(rho):
    gamma = 7
    c_0 = 1
    # TODO: maybe refine reference density, depending on what it turns
    #       out to be
    rho_0 = 1.1
    # TODO: find out what p_B actually is
    p_B = 0

    return (((c_0**2)*rho_0/gamma) * (
        ((rho/rho_0)**gamma)-1) + p_B)



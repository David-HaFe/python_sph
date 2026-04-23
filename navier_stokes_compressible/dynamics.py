# navier stokes equations, compressible as well as incompressible

import numpy as np
import sys

from utils.diagnostics import diagnostics
from kernels.wendland import wendland, gradient_W, normalised_gradient_W
from kernels.gauss import gauss, nabla, laplace
from config import (
    no_particles,
    masses,
    gravity,
)


# implements the discretized navier stokes equation for the ode solver
# y consists of the triple [positions, velocities, density] for each particle,
# all concatenated like this one after the other
# this is the COMPRESSIBLE case
def dynamics(t, y, is_border_particle):
    diagnostics.time_dynamics()

    y_dot = np.zeros(np.size(y))
    r_dot = np.zeros((no_particles, 2))
    v_dot = np.zeros((no_particles, 2))
    rho_dot = np.zeros(no_particles)

    # TODO: these have to be updated, not sure how to do that
    m = masses
    mu = np.full(no_particles, 0.1)

    eta = 0.01

    r = y[: 2 * no_particles]
    v = y[2 * no_particles : 4 * no_particles]
    rho = y[4 * no_particles :]
    r = r.reshape(-1, 2)
    v = v.reshape(-1, 2)

    for i, (r_i, v_i, rho_i) in enumerate(zip(r, v, rho)):
        if not is_border_particle[i]:
            # reset everything and get new vectors
            pressure_term = np.zeros(2)
            viscosity_term = np.zeros(2)
            rho_dot_i = 0

            p_i = calculate_pressure(rho_i)

            for j, (r_j, v_j, rho_j) in enumerate(zip(r, v, rho)):
                # x_b = np.array(x_b)
                particle_close_enough = wendland(r_i, r_j) > 0

                # if particle_close_enough:
                if particle_close_enough and not (i == j):

                    # calculate dv/dt
                    # v_b = np.array(v_b)
                    p_j = calculate_pressure(rho_j)

                    pressure_term += (
                        m[j]
                        * (p_j / np.power(rho_j, 2) + p_i / np.power(rho_i, 2))
                        * gradient_W(r_i, r_j)
                    )

                    delta_r = r_i - r_j
                    delta_v = v_i - v_j

                    viscosity_term += (
                        m[j]
                        * (
                            (mu[i] + mu[j])
                            / (rho_i * rho_j)
                            * np.dot(delta_r, delta_v)
                            / (np.dot(delta_r, delta_r) + 0.01 * eta**2)
                        )
                        * gradient_W(r_i, r_j)
                    )

                    # calculate d(rho)/dt
                    rho_dot_i += (m[j] / rho_j) * (
                        np.dot(v_j - v_i, gradient_W(r_i, r_j))
                    )

            # fill solution array
            r_dot[i] = v_i
            v_dot[i] = gravity - pressure_term + viscosity_term
            rho_dot[i] = -rho_i * rho_dot_i
        else:
            r_dot[i] = np.zeros(2)
            v_dot[i] = np.zeros(2)
            rho_dot[i] = 0

    r_dot = r_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    y_dot = np.concatenate((r_dot, v_dot, rho_dot))

    sys.stdout.write(f"\r\033[Ksimulating @ {t}")
    sys.stdout.flush()

    diagnostics.time_dynamics()
    return y_dot


# calculates the pressure based on given formula
def calculate_pressure(rho):
    gamma = 7
    c_0 = 10
    # TODO: maybe refine reference density, depending on what it turns
    #       out to be
    rho_0 = 1
    # TODO: find out what p_B actually is
    p_B = 0

    return ((c_0**2) * rho_0 / gamma) * (((rho / rho_0) ** gamma) - 1) + p_B

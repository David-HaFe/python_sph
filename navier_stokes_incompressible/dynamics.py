# navier stokes equations, compressible as well as incompressible

import numpy as np
import sys

from utils.diagnostics import diagnostics
from kernels.wendland import wendland, gradient_W, normalised_gradient_W
from kernels.gauss import gauss, nabla, laplace
from config import (
    no_particles,
    gravity,
    kinetic_viscosities,
)


# implements the discretized navier stokes equation for the ode solver
# y consists of the triple [positions, velocities, density] for each particle,
# all concatenated like this one after the other
# this is the INCOMPRESSIBLE case
def dynamics(t, y, is_border_particle):
    diagnostics.time_dynamics()
    y_dot = np.zeros(np.size(y))
    r_dot = np.zeros((no_particles, 2))
    v_dot = np.zeros((no_particles, 2))
    p_dot = np.zeros((no_particles, 1))

    r = y[: 2 * no_particles]
    v = y[2 * no_particles : 4 * no_particles]
    p = y[4 * no_particles :]
    r = r.reshape(-1, 2)
    v = v.reshape(-1, 2)
    p = p.reshape(-1, 1)

    nu = kinetic_viscosities

    for i, (r_i, v_i) in enumerate(zip(r, v)):
        if not is_border_particle[i]:
            # reset everything and get new vectors
            acceleration_i = (
                # nu[i]*laplace(r_i, v_i, r, v)
                gravity
                + nu[i] * laplace(r_i, v_i, r, v)
            )
            r_dot[i] = v_i
            v_dot[i] = acceleration_i
        else:
            r_dot[i] = np.zeros(2)
            v_dot[i] = np.zeros(2)

        # don't update pressure here, just push through
        p_dot[i] = np.zeros(1)

    r_dot = r_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    p_dot = p_dot.reshape(-1, order="C")

    y_dot = np.concatenate((r_dot, v_dot, p_dot))

    sys.stdout.write(f"\r\033[Ksimulating @ {t}")
    sys.stdout.flush()

    diagnostics.time_dynamics()
    return y_dot



import numpy as np

from utils.diagnostics import diagnostics
from config import model_parameters
from kernels.wendland import wendland, normalised_gradient_W

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
                particle_close_enough = wendland(x_a, x_b) > 0

                # particle is close enough
                if (particle_close_enough and not (a == b)):
                    divergence_v += (m[b]/rho[b])*(
                        (v_b - v_a)
                    )

            delta_W_ab_norm = normalised_gradient_W(x_a, x_b, x)
            divergence_v = divergence_v.reshape(-1, 2)
            divergence_v = np.dot(divergence_v, delta_W_ab_norm)

            # TODO: calculate delta P here
            delta_P_plus = 0

            # calculate new velocity (forces already added in step 1
            v_dot[a] -= dt*(delta_P_plus/rho[a])

    x_dot = x_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    y_dot = np.concatenate((x_dot, v_dot))

    diagnostics.time_poisson()
    return y_dot



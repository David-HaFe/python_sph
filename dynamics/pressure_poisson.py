

import numpy as np

from utils.diagnostics import diagnostics
from config import model_parameters
from kernels.wendland import wendland, normalised_gradient_W
from kernels.gauss import gauss, nabla, laplace

# evaluate the poisson pressure equations for given state
def poisson_pressure_equation(t, y, is_border_particle, dt):
    diagnostics.time_poisson()
    no_particles = np.size(y)//(2+2)
    delta_p = np.empty(no_particles)

    r_dot = np.zeros((no_particles, 2))
    v_dot = np.zeros((no_particles, 2))

    r = y[: 2*no_particles]
    r = r.reshape(-1, 2)
    v = y[2*no_particles :]
    v = v.reshape(-1, 2)

    m = model_parameters.m
    rho = model_parameters.rho

    b = []
    for i, (r_i, v_i) in enumerate(zip(r, v)):
        if not is_border_particle[i]:
            # set up b vector
            b.extend([nabla(r_i, v_i, r, v)/dt])
            b = np.array(b, dtype=float)

    A = _assemble_LHS(r, dt)

    sol = np.linalg.solve(A, b)

    r_dot = r_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    y_dot = np.concatenate((r_dot, v_dot))

    diagnostics.time_poisson()
    return y_dot

# construct A matrix for PPE
def _assemble_LHS(r, dt):
    no_particles = np.size(r)//2
    A = np.zeros((no_particles, no_particles))

    for i, (r_i, v_i) in enumerate(zip(r, v)):
        for j, (r_j, v_j) in enumerate(zip(r, v)):
            kernel = gauss(r_i, r_j)

            if kernel > 0 and not i == j:
                weight = gauss(r_i, r_j, h)
                A[i, j] += weight
                A[i, i] -= weight

    return A



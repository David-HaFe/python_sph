# main file for simulating COMPRESSIBLE fluid with navier stokes equations
# author: David Hambach Ferrer

import numpy as np
from scipy.integrate import solve_ivp
import random as rnd

from solvers.chorin import chorin
import navier_stokes_compressible.dynamics as navier_stokes_compressible
from utils.diagnostics import diagnostics
from utils.generate_border import generate_border
from config import (
    x_positions,
    y_positions,
    no_particles,
    t0,
    t1,
    no_steps,
)


def main():
    def noise():
        return -0.1 + 0.2 * rnd.random()

    r_0 = []
    v_0 = []
    rho_0 = []
    is_border_particle = []

    for _, y in enumerate(y_positions):
        for _, x in enumerate(x_positions):
            r_0.extend([x, y])
            v_0.extend([0, 0])
            rho_0.extend([1])

            # add wall particle flag
            is_border_particle.extend([False])

    r_0, rho_0, rho_0, is_border_particle = generate_border(
        r_0,
        rho_0,
        rho_0,
        is_border_particle,
    )

    r_0 = np.array(r_0, dtype=float)
    v_0 = np.array(v_0, dtype=float)
    rho_0 = np.array(rho_0, dtype=float)
    y_0 = np.concatenate((r_0, v_0, rho_0))
    is_border_particle = np.array(is_border_particle)

    # simulation
    diagnostics.time_ode()

    # sol = euler_forward(
    #     function=navier_stokes,
    #     initial_condition=y_0,
    #     t_start=0,
    #     t_end=4,
    #     dt=.01,
    # )

    # solver setup
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=no_steps)

    # sol = chorin(
    #     forward_equation=lambda t, y: navier_stokes_incompressible(
    #         t,
    #         y,
    #         is_border_particle,
    #     ),
    #     projection_equation=lambda t, y: poisson_pressure_equation(
    #         t,
    #         y,
    #         is_border_particle,
    #         dt,
    #     ),
    #     initial_condition=y_0,
    #     t_start=t_0,
    #     t_end=t_1,
    #     dt=.01,
    # )

    sol = solve_ivp(
        fun=lambda t, y: navier_stokes_compressible.dynamics(
            t,
            y,
            is_border_particle,
        ),
        t_span=t_span,
        y0=y_0,
        method="RK23",
        rtol=1e-3,
        atol=1e-3,
        t_eval=t_eval,
    )

    diagnostics.time_ode()

    x = sol.y[0 : 2 * no_particles : 2, :]
    y = sol.y[1 : 2 * no_particles : 2, :]

    t = sol.t

    return t, x, y, is_border_particle

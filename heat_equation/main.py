# main file for the implementation of the heat equations
# author: David Hambach Ferrer

import numpy as np
from scipy.integrate import solve_ivp
from math import sin, cos, pi
from random import uniform, seed
from tqdm import tqdm

import heat_equation.dynamics as heat_equation
import heat_equation_analytical.dynamics as heat_equation_analytical

from utils.diagnostics import diagnostics
from kernels.gauss import gauss
from config import (
    sim_result,
    x_positions,
    y_positions,
    no_particles,
    t0,
    t1,
    no_steps,
    border,
)
from utils.generate_border import generate_border

seed(37)


def noise(magnitude):
    return uniform(-magnitude, magnitude)


def main():
    r_0 = []
    T_0 = []
    is_border_particle = []

    # fill initial condition
    for _, y in enumerate(y_positions):
        for _, x in enumerate(x_positions):
            r_0.extend([x, y])
            # T_0.extend([0.1 * cos(8 * pi * x) + noise(0.05)])
            # T_0.extend([noise(0.1)])
            # T_0.extend([0.3 * sin(14 * x) + 0.3 * cos(14 * y) + noise(0.03)])
            T_0.extend([heat_equation_analytical.dynamics(t0, x, y) + noise(0.01)])
            # T_0.extend([5 * gauss(np.zeros(2), np.array([x, y]), 1.5 * border)])
            is_border_particle.extend([False])

    base_temp = [0]
    # this also requires and returns a pressure for the navier stokes equations,
    # just pass through and ignore afterwards
    dummy_pressure = []
    r_0, T_0, _, is_border_particle = generate_border(
        r_0, T_0, dummy_pressure, is_border_particle
    )

    r_0 = np.array(r_0, dtype=float)
    T_0 = np.array(T_0, dtype=float)
    is_border_particle = np.array(is_border_particle)

    y_0 = np.concatenate((r_0, T_0))

    # solver setup
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=no_steps)

    # solve
    pbar = 0
    # with tqdm(total=t_span[1], unit="t") as pbar:
    diagnostics.time_ode()
    sol = solve_ivp(
        fun=lambda t, y: heat_equation.dynamics(
            t,
            y,
            is_border_particle,
            pbar,
            t_eval,
        ),
        t_span=t_span,
        y0=y_0,
        method="RK23",
        rtol=1e-3,
        atol=1e-3,
        t_eval=t_eval,
    )
    diagnostics.time_ode()

    t = sol.t
    x = sol.y[0 : 2 * no_particles : 2, :].T
    y = sol.y[1 : 2 * no_particles : 2, :].T
    T = sol.y[2 * no_particles :, :].T
    data_2_dummy = np.zeros((no_steps, no_particles))

    result = sim_result(
        t=t,
        x=x,
        y=y,
        data_1=T,
        data_2=data_2_dummy,
        is_border_particle=is_border_particle,
    )

    return result

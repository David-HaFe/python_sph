# heat equation using the exact variables

import numpy as np
from itertools import product

import heat_equation_analytical.dynamics as heat_equation_analytical
from config import (
    t0,
    t1,
    no_steps,
    no_particles,
    x_positions,
    y_positions,
)


def main():
    times = np.linspace(t0, t1, no_steps)

    x_sol = np.zeros((no_particles, no_steps))
    y_sol = np.zeros((no_particles, no_steps))
    T_sol = np.zeros((no_particles, no_steps))

    for t_index, t in enumerate(times):
        for particle_index, (x, y) in enumerate(product(x_positions, y_positions)):
            x_sol[particle_index][t_index] = float(x)
            y_sol[particle_index][t_index] = float(y)
            T_sol[particle_index][t_index] = float(
                heat_equation_analytical.dynamics(t, x, y)
            )

    times = times.tolist()
    x_sol = x_sol.tolist()
    y_sol = y_sol.tolist()
    T_sol = T_sol.tolist()
    is_border_particle = np.full(no_particles, False)

    return times, x_sol, y_sol, T_sol, is_border_particle

# code for the solution function
#
# T(t, x, y) = cos(a · √(x² + y²) )

import numpy as np
from itertools import product

from config import heat_alpha
from config import (
    sim_result,
    t0,
    t1,
    no_steps,
    no_particles,
    x_positions,
    y_positions,
    heat_alpha,
)

scaling = 3
def main():
    times = np.linspace(t0, t1, no_steps)

    x_sol = np.zeros((no_steps, no_particles))
    y_sol = np.zeros((no_steps, no_particles))
    T_sol = np.zeros((no_steps, no_particles))

    for t_index, t in enumerate(times):
        for particle_index, (x, y) in enumerate(product(x_positions, y_positions)):
            x_sol[t_index][particle_index] = float(x)
            y_sol[t_index][particle_index] = float(y)
            T_sol[t_index][particle_index] = float(solution(x, y, t))

    data_2_dummy = np.zeros((no_steps, no_particles))
    is_border_particle = np.full(no_particles, False)

    result = sim_result(
        t=times,
        x=x_sol,
        y=y_sol,
        data_1=T_sol,
        data_2=data_2_dummy,
        is_border_particle=is_border_particle,
    )
    return result


def solution(x: float, y: float, t: float = 0):
    return np.cos(scaling * np.sqrt(x**2 + y**2))


def source_term_heat_equation(t: float, x: float, y: float):
    norm = np.linalg.norm([x, y])
    return heat_alpha * (- scaling / norm * np.sin(scaling * norm) - scaling**2 * np.cos(scaling * norm))

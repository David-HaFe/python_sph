

# code for the solution function
#
# T(t, x, y) = (x² + y²) cos(t)

import numpy as np
from itertools import product

import manufactured_solutions.solution_1.dynamics.solution as manufactured_solution
from config import (
    sim_result,
    t0,
    t1,
    no_steps,
    no_particles,
    x_positions,
    y_positions,
)

def main():
    times = np.linspace(t0, t1, no_steps)

    x_sol = np.zeros((no_steps, no_particles))
    y_sol = np.zeros((no_steps, no_particles))
    T_sol = np.zeros((no_steps, no_particles))

    for t_index, t in enumerate(times):
        for particle_index, (x, y) in enumerate(product(x_positions, y_positions)):
            x_sol[t_index][particle_index] = float(x)
            y_sol[t_index][particle_index] = float(y)
            T_sol[t_index][particle_index] = float(
                manufactured_solution.solution(t, x, y)
            )

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



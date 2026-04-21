# heat equation using the exact variables

import numpy as np
from itertools import product
from playsound3 import playsound

from dynamics.heat_equation_analytical import heat_equation_analytical
from utils.heat_map import heat_plot
from config import (
    t0,
    t1,
    dt,
    no_steps,
    no_particles,
    x_positions,
    y_positions,
)

times = np.linspace(t0, t1, no_steps)

x_sol = np.zeros((no_particles, no_steps))
y_sol = np.zeros((no_particles, no_steps))
T_sol = np.zeros((no_particles, no_steps))

for t_index, t in enumerate(times):
    for particle_index, (x, y) in enumerate(product(x_positions, y_positions)):
        x_sol[particle_index][t_index] = x
        y_sol[particle_index][t_index] = y
        T_sol[particle_index][t_index] = heat_equation_analytical(t, x, y)

print(times.shape)
print(x_sol.shape)
print(y_sol.shape)
print(T_sol.shape)
heat_plot(times, x_sol, y_sol, T_sol)

# ready
playsound("media/ding.wav")

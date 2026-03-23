

# main file for the implementation of sph
# author: David Hambach Ferrer

# import numpy as np
import numpy as np
import jax.numpy as jnp
import matplotlib as plt
import cProfile
import pstats
import io
from math import floor

from integrator import solve
from equations import navier_stokes
from diagnostics import diagnostics
from animation import animate_solution

# initialize grid
initial_condition = np.array([])

for x in range(0, 3):
    for y in range(0, 3):
        # position
        initial_condition = np.append(initial_condition, np.array([x, y]))

        # velocity
        initial_condition = np.append(initial_condition, np.array([0, 0]))

        # initial density
        initial_condition = np.append(initial_condition, 1)

# simulation
diagnostics.time_ode()
sol = solve(
    function=navier_stokes,
    initial_condition=initial_condition,
    dt=.05,
)
# sol = solve_ivp(
#     fun = navier_stokes,
#     t_span = [0, 1],
#     y0 = initial_condition,
#     method = 'RK45',
#     rtol = 1e-3,
#     atol = 1e-5,
#     dense_output = False,
# )
diagnostics.time_ode()

diagnostics.print_diagnostics()
animate_solution(sol.t, sol.y)



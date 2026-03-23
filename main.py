

# main file for the implementation of sph
# author: David Hambach Ferrer

# import numpy as np
import numpy as np
import jax.numpy as jnp
from scipy.integrate import odeint
import matplotlib as plt
import cProfile
import pstats
import io
from math import floor

from equations import navier_stokes
from diagnostics import diagnostics
from animation import animate_solution

# initialize grid
initial_condition = np.array([])

for x in range(0, 4):
    for y in range(0,4):
        # position
        initial_condition = np.append(initial_condition, np.array([.1*x, .1*y]))

        # velocity
        initial_condition = np.append(initial_condition, np.array([0, 0]))

        # initial density
        initial_condition = np.append(initial_condition, 1)

# simulation
t = np.linspace(0,1,20)
diagnostics.time_ode()
solution = odeint(
    navier_stokes,
    initial_condition,
    t,
)
diagnostics.time_ode()

diagnostics.print_diagnostics()
animate_solution(t, solution)



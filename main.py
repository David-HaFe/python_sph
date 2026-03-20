

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

from equations import navier_stokes
from diagnostics import diagnostics
from animation import animate_solution

# constants
rho_a = 1
gravity = np.array([0, -9.81])
no_particles = 4

# initialize grid
initial_condition = np.array([])

for i in range(0, 4):
    initial_condition = np.append(initial_condition, jnp.array([i, i]))
    initial_condition = np.append(initial_condition, jnp.array([-i, -i]))

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



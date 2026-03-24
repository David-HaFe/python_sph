

# main file for the implementation of sph
# author: David Hambach Ferrer

# import numpy as np
import numpy as np

from integrator import solve
from equations import navier_stokes
from diagnostics import diagnostics
from animation import animate_solution

# initialize grid
initial_condition = np.array([])
is_wall_particle = np.array([])

x_limit = 2
y_limit = 2

for x in range(0, x_limit):
    for y in range(0, y_limit):
        # position
        initial_condition = np.append(initial_condition, np.array([x, y]))

        # velocity
        initial_condition = np.append(initial_condition, np.array([0, 0]))

        # initial density
        initial_condition = np.append(initial_condition, 1)

        is_wall_particle = np.append(is_wall_particle, False)

dummy_density = 100
# initialize wall particles
for x in range(-1, x_limit + 1):
    # left wall
    initial_condition = np.append(initial_condition, np.array([x, -1]))
    # dummy values for velocity and density
    initial_condition = np.append(
        initial_condition, np.array([0, 0, dummy_density]))
    is_wall_particle = np.append(is_wall_particle, True)

    # right wall
    initial_condition = np.append(initial_condition, np.array([x, y_limit+1]))
    # dummy values for velocity and density
    initial_condition = np.append(
        initial_condition, np.array([0, 0, dummy_density]))
    is_wall_particle = np.append(is_wall_particle, True)

for y in range(-1, y_limit+1):
    # bottom wall
    initial_condition = np.append(initial_condition, np.array([-1, y]))
    # dummy values for velocity and density
    initial_condition = np.append(
        initial_condition, np.array([0, 0, dummy_density]))
    is_wall_particle = np.append(is_wall_particle, True)

# simulation
diagnostics.time_ode()
sol = solve(
    function=lambda t, q: navier_stokes(t, q, is_wall_particle),
    initial_condition=initial_condition,
    dt=.01,
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



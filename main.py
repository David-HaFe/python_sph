# main file for simulating incompressible fluid with navier stokes equations
# author: David Hambach Ferrer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import random as rnd

from solver.chorin import chorin
from solver.euler_forward import euler_forward
from dynamics.navier_stokes import navier_stokes_incompressible
from dynamics.pressure_poisson import poisson_pressure_equation
from utils.diagnostics import diagnostics
from utils.particle_positions import particle_positions
from config import model_parameters, x_limit, y_limit
from initial_condition.generate_border import generate_border

# initialize grid
initial_condition = np.array([])

spacing = 1
wall_spacing = 0.1
no_particles = x_limit * y_limit


def noise():
    return -0.1 + 0.2 * rnd.random()


r_0 = []
v_0 = []
p_0 = []
is_border_particle = []

for x in range(0, x_limit):
    for y in range(0, y_limit):
        r_0.extend([x * spacing, y * spacing])

        if x == 0:
            x_vel = 1
        elif x == x_limit - 1:
            x_vel = -1
        else:
            x_vel = 0

        v_0.extend([x_vel, 0])
        p_0.extend([1])

        # add wall particle flag
        is_border_particle.extend([False])

border_velocity = [0, 0]
r_0, v_0, p_0, is_border_particle, no_particles = generate_border(
    r_0, v_0, p_0, border_velocity, is_border_particle, no_particles
)

r_0 = np.array(r_0, dtype=float)
v_0 = np.array(v_0, dtype=float)
p_0 = np.array(p_0, dtype=float)
y_0 = np.concatenate((r_0, v_0, p_0))
is_border_particle = np.array(is_border_particle)

model_parameters.set_no_particles(np.size(y_0) // (2 + 2 + 1))

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
t_0 = 0
t_1 = 3
t_span = (t_0, t_1)
steps = 10 * t_1
t_eval = np.linspace(t_0, t_1, num=steps)
dt = (t_1 - t_0) / steps

sol = chorin(
    forward_equation=lambda t, y: navier_stokes_incompressible(
        t,
        y,
        is_border_particle,
    ),
    projection_equation=lambda t, y: poisson_pressure_equation(
        t,
        y,
        dt,
        is_border_particle,
    ),
    initial_condition=y_0,
    t_start=t_0,
    t_end=t_1,
    dt=0.01,
)

# sol = solve_ivp(
#     fun=lambda t, y: navier_stokes_compressible(
#         t,
#         y,
#         is_wall_particle,
#     ),
#     t_span = t_span,
#     y0 = y_0,
#     method = "RK23",
#     rtol = 1e-3,
#     atol = 1e-3,
#     t_eval = t_eval,
# )
# print("")
# print(sol.message)

diagnostics.time_ode()

x = sol.y[0 : 2 * no_particles : 2, :]
y = sol.y[1 : 2 * no_particles : 2, :]

t = sol.t
particle_positions(t, x, y, is_border_particle)

print("")
diagnostics.print_diagnostics()

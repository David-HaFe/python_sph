# main file for the implementation of the heat equations
# author: David Hambach Ferrer

import numpy as np
from scipy.integrate import solve_ivp
import csv
from playsound3 import playsound
from tqdm import tqdm

from dynamics.heat_equation import heat_equation
from utils.diagnostics import diagnostics
from kernels.gauss import gauss
from utils.heat_map import heat_plot
from utils.particle_positions import particle_positions
from dynamics.heat_equation_analytical import heat_equation_analytical
from config import (
    no_particles_x,
    no_particles_y,
    x_positions,
    y_positions,
    no_particles,
    border,
    t0,
    t1,
    no_steps,
    dt,
)
from initial_condition.generate_border import generate_border
from utils.save_solution import save_run

# initialize grid
initial_condition = np.array([])
spacing = 0.1

r_0 = []
T_0 = []
is_border_particle = []

# with open("initial_condition/initial_condition.csv") as file:
#     reader = csv.reader(file)
#     initial_temps = list(reader)

# fill initial condition
for _, y in enumerate(y_positions):
    for _, x in enumerate(x_positions):
        r_0.extend([x, y])
        T_0.extend([heat_equation_analytical(t0, x, y)])
        # T_0.extend([
        #     5 * gauss(np.zeros(2), np.array([x, y]), 1.5 * border)
        # ])
        is_border_particle.extend([False])

base_temp = [0]
# this also requires and returns a pressure for the navier stokes equations,
# just pass through and ignore afterwards
dummy_pressure = []
r_0, T_0, dummy_pressure, is_border_particle = generate_border(
    r_0, T_0, dummy_pressure, base_temp, is_border_particle
)

r_0 = np.array(r_0, dtype=float)
T_0 = np.array(T_0, dtype=float)
is_border_particle = np.array(is_border_particle)

y_0 = np.concatenate((r_0, T_0))

# solver setup
t_span = (t0, t1)
t_eval = np.linspace(t0, t1, num=no_steps)

# solve
diagnostics.time_ode()
sol = solve_ivp(
    fun=lambda t, y: heat_equation(t, y, is_border_particle),
    t_span=t_span,
    y0=y_0,
    method="RK23",
    rtol=1e-3,
    atol=1e-3,
    t_eval=t_eval,
)
diagnostics.time_ode()

x = sol.y[0 : 2 * no_particles : 2, :]
y = sol.y[1 : 2 * no_particles : 2, :]
T = sol.y[2 * no_particles :, :]

t = sol.t

heat_plot(t, x, y, T)
particle_positions(t, x, y, is_border_particle)

print("")

save_run(t, T, "heat_eq")

diagnostics.print_diagnostics()

# ready
playsound("media/ding.wav")

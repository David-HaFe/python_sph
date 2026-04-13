

# main file for the implementation of the heat equations
# author: David Hambach Ferrer

import numpy as np
from scipy.integrate import solve_ivp
import csv

from dynamics.heat_equation import heat_equation
from utils.diagnostics import diagnostics
from kernels.gauss import gauss
from utils.heat_map import heat_plot
from utils.particle_positions import particle_positions
from config import x_limit, y_limit
from initial_condition.generate_border import generate_border

# initialize grid
initial_condition = np.array([])
spacing = .1

no_particles = x_limit*y_limit

r_0 = []
T_0 = []
is_border_particle = []

# with open("initial_condition/initial_condition.csv") as file:
#     reader = csv.reader(file)
#     initial_temps = list(reader)

# fill initial condition
for y in range(0, y_limit):
    for x in range(0, x_limit):
        r_0.extend([x, y])
        T_0.extend([5*gauss(np.array([x,y]), np.array([x_limit/2,y_limit/2]))])
        is_border_particle.extend([False])

base_temp = [0]
r_0, T_0, is_border_particle, no_particles = generate_border(
    r_0, T_0, base_temp, is_border_particle, no_particles
)

r_0 = np.array(r_0, dtype=float)
T_0 = np.array(T_0, dtype=float)
is_border_particle = np.array(is_border_particle)

y_0 = np.concatenate((r_0, T_0))

# solver setup
t_0 = 0
t_1 = 30
t_span = (t_0, t_1)
steps = 10*t_1
t_eval = np.linspace(t_0, t_1, num=steps)
dt = (t_1-t_0)/steps

# solve
diagnostics.time_ode()
sol = solve_ivp(
    fun=lambda t, y: heat_equation(t, y, is_border_particle),
    t_span=t_span,
    y0 = y_0,
    method = "RK23",
    rtol = 1e-3,
    atol = 1e-3,
    t_eval = t_eval,
)
diagnostics.time_ode()

x = sol.y[0:2*no_particles:2, :]
y = sol.y[1:2*no_particles:2, :]
T = sol.y[2*no_particles:, :]

t = sol.t
heat_plot(t, x, y, T)
particle_positions(t, x, y, is_border_particle)

print("")
diagnostics.print_diagnostics()





# main file for the implementation of the heat equations
# author: David Hambach Ferrer

import numpy as np
from scipy.integrate import solve_ivp

from dynamics.heat_equation import heat_equation
from utils.diagnostics import diagnostics
from utils.heat_map import heat_plot
from kernels.gauss import gauss

# initialize grid
initial_condition = np.array([])
x_limit = 20
y_limit = 20
spacing = .1

no_particles = x_limit*y_limit

r_0 = []
T_0 = []
is_border_particle = []

# fill initial condition
for x in range(0, x_limit):
    for y in range(0, y_limit):
        r_0.extend([x, y])
        T_0.extend([gauss(np.array([x_limit/2,y_limit/2]), np.array([x,y]), (x_limit+y_limit)/4)])
        is_border_particle.extend([False])

base_temp=0
for x in range (-1, x_limit+1):
    r_0.extend([x, -1])
    T_0.extend([base_temp])
    is_border_particle.extend([True])

    r_0.extend([x, y_limit+1])
    T_0.extend([base_temp])
    is_border_particle.extend([True])
    no_particles += 2

for y in range (0, y_limit):
    r_0.extend([-1, y])
    T_0.extend([base_temp])
    is_border_particle.extend([True])

    r_0.extend([x_limit+1, y])
    T_0.extend([base_temp])
    is_border_particle.extend([True])
    no_particles += 2

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

print("")
diagnostics.print_diagnostics()

x = sol.y[0:2*no_particles:2, :]
y = sol.y[1:2*no_particles:2, :]
T = sol.y[2*no_particles:, :]

t = sol.t
heat_plot(t, x, y, T)



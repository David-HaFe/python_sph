

# main file for the implementation of the heat equations
# author: David Hambach Ferrer

import numpy as np
from scipy.integrate import solve_ivp

from dynamics.heat_equation import heat_equation
from utils.diagnostics import diagnostics
from utils.heat_map import heat_plot

# initialize grid
initial_condition = np.array([])
x_limit = 10
y_limit = 10
spacing = .1

no_particles = x_limit*y_limit

r_0 = []
T_0 = []

# fill initial condition
for x in range(0,x_limit):
    for y in range(0,y_limit):
        r_0.extend([x,y])
        T_0.extend([x+y])

r_0 = np.array(r_0, dtype=float)
T_0 = np.array(T_0, dtype=float)

y_0 = np.concatenate((r_0, T_0))

# solver setup
t_0 = 0
t_1 = 5
t_span = (t_0, t_1)
steps = 100
t_eval = np.linspace(t_0, t_1, num=steps)
dt = (t_1-t_0)/steps

# solve
diagnostics.time_ode()
sol = solve_ivp(
    fun=heat_equation,
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

r = sol.y[: 2*no_particles]
T = sol.y[2*no_particles :]
t = sol.t
heat_plot(t, r, T)



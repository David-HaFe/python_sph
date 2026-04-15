

import numpy as np
from types import SimpleNamespace
from utils.diagnostics import diagnostics

def chorin(
    forward_equation,
    projection_equation,
    initial_condition,
    t_start,
    t_end,
    dt=.01,
):
    # set up solution array
    no_iterations = int((t_end-t_start)/dt)
    solution = np.empty((np.size(initial_condition), no_iterations+1))
    times = np.empty(no_iterations+1)

    # write initial condition to state
    t = t_start
    y = initial_condition

    solution[:, 0] = initial_condition
    times[0] = t_start

    for i in range(1, no_iterations+1):
        # intermediate step
        y = y + dt * forward_equation(t, y)

        diagnostics.log_np_array(y)
        # poisson pressure equation
        y = y + dt * projection_equation(t, y)
        t += dt
        times[i] = t
        solution[:, i] = y

    return SimpleNamespace(t=times, y=solution)



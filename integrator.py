

# simple integrator for sph implementation

import numpy as np
from types import SimpleNamespace

def solve(
    function,
    initial_condition,
    t_start=0,
    t_end=1,
    dt=.01,
):
    no_iterations = int(t_end/dt)
    solution = np.empty((np.size(initial_condition), no_iterations))
    y = initial_condition
    t = t_start
    times = np.array(t_start)

    for i in range(0, no_iterations):
        solution[:,i] = y
        y = y + dt*function(t, y)
        t += dt
        times = np.append(times, t)

    print(solution)

    return SimpleNamespace(t=times, y=solution)



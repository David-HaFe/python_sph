

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
    solution = np.empty((np.size(initial_condition), no_iterations+1))
    y = initial_condition
    t = t_start
    times = np.empty(no_iterations+1)
    solution[:,0] = initial_condition
    times[0] = t_start

    for i in range(1, no_iterations+1):
        y = y + dt*function(t, y)
        t += dt
        solution[:,i] = y
        times[i] = t

    print("iterations: " + str(no_iterations))

    return SimpleNamespace(t=times, y=solution)



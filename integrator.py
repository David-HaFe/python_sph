

# simple integrator for sph implementation

import numpy as np
from types import SimpleNamespace

def euler_forward(
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

    print("")
    print("iterations: " + str(no_iterations))

    return SimpleNamespace(t=times, y=solution)

def chorin(
    forward_equation,
    projection_equation,
    initial_condition,
    t_start=0,
    t_end=0,
    dt=.01,
):
    # set up solution array
    no_iterations = int(t_end/dt)
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

        # poisson pressure equation
        # y = y - dt * projection_equation(t, y)
        t += dt
        times[i] = t
        solution[:, i] = y

    print("")
    print("iterations: " + str(no_iterations))
    return SimpleNamespace(t=times, y=solution)



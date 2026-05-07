import numpy as np
from types import SimpleNamespace


def ruku_4(
    function,
    initial_condition,
    t_start=0,
    t_end=1,
    dt=0.01,
):
    # calculate stuff
    no_iterations = int((t_end - t_start) / dt)
    times = np.linspace(t_start, t_end, no_iterations)
    solution = np.empty((np.size(initial_condition), np.size(times)))

    # set inital condition as starting value
    y = initial_condition

    # save initial condition
    solution[:, 0] = initial_condition

    # I want to start my indices at 1 (:
    k = np.empty((np.size(initial_condition), 5))
    half_dt = dt * 0.5

    # iterate up to the last entry of times, not including it, and also start
    # walking index at 1
    for index, time in enumerate(times[: -1], start=1):
        k[:, 1] = function(time, y)
        k[:, 2] = function(time + half_dt, y + half_dt * k[:, 1])
        k[:, 3] = function(time + half_dt, y + half_dt * k[:, 2])
        k[:, 4] = function(time + dt, y + dt * k[:, 3])

        # final answer
        y += (dt / 6) * (k[:, 1] + 2 * k[:, 2] + 2 * k[:, 3] + k[:, 4])
        solution[:, index] = y

    return SimpleNamespace(t=times, y=solution)

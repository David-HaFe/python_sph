import numpy as np
from types import SimpleNamespace
from utils.diagnostics import diagnostics
import sys


def ruku_4(
    dynamics,
    border_update,
    initial_condition,
    t_start=0,
    t_end=1,
    dt=0.01,
):
    # calculate stuff
    no_iterations = int((t_end - t_start) / dt)
    times = np.linspace(t_start, t_end, no_iterations)
    solution = np.zeros((np.size(initial_condition), np.size(times)))

    # set inital condition as starting value
    y = initial_condition

    # save initial condition
    solution[:, 0] = initial_condition

    # I want to start my indices at 1 (:
    k = np.empty((np.size(initial_condition), 5))
    half_dt = dt * 0.5

    diagnostics.log_full_np_array(times)
    # iterate up to the last entry of times, not including it, and also start
    # walking index at 1
    try:
        for index, time in enumerate(times[:-1], start=1):
            sys.stdout.write(f"\r\033[Ksimulating @ {time}")
            sys.stdout.flush()

            k[:, 1] = dynamics(time, y)
            k[:, 2] = dynamics(time + half_dt, y + half_dt * k[:, 1])
            k[:, 3] = dynamics(time + half_dt, y + half_dt * k[:, 2])
            k[:, 4] = dynamics(time + dt, y + dt * k[:, 3])

            # final answer
            y += (dt / 6) * (k[:, 1] + 2 * k[:, 2] + 2 * k[:, 3] + k[:, 4])
            y = solution[:, index] = y

            border_update(time + dt, y)
    except np.linalg.LinAlgError:
        print("\nLINALG ERROR - exiting now")
        pass

    return SimpleNamespace(t=times, y=solution)

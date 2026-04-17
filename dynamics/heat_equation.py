

# heat equations

import numpy as np
import sys

from utils.diagnostics import diagnostics
from kernels.gauss import gauss, nabla, laplace

def heat_equation(t, y, is_border_particle):
    diagnostics.time_dynamics()

    alpha = .3

    no_particles = np.size(y)//(2+1)
    y_dot = np.zeros(np.size(y))
    r_dot = np.zeros((no_particles, 2))
    T_dot = np.zeros((no_particles, 1))

    r = y[: 2*no_particles]
    T = y[2*no_particles :]
    r = r.reshape(-1, 2)
    T = T.reshape(-1, 1)

    for a, (r_a, T_a) in enumerate(zip(r, T)):
        if not is_border_particle[a]:
            temperature_diff = alpha*laplace(r_a, T_a, r, T)

            r_dot[a] = np.zeros(2)
            T_dot[a] = temperature_diff
        else:
            r_dot[a] = np.zeros(2)
            T_dot[a] = np.zeros(1)
            # T_dot[a] = np.array([np.sin(t)])

    r_dot = r_dot.reshape(-1, order="C")
    T_dot = T_dot.reshape(-1, order="C")
    y_dot = np.concatenate((r_dot, T_dot))

    diagnostics.log_np_array(T_dot)

    sys.stdout.write(f"\r\033[Ksimulating @ {t}")
    sys.stdout.flush()

    diagnostics.time_dynamics()
    return y_dot



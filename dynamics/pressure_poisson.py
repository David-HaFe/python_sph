

import numpy as np

from utils.diagnostics import diagnostics
from config import model_parameters
from kernels.gauss import gauss, nabla, laplace

# evaluate the poisson pressure equations for given state
def poisson_pressure_equation(t, y, dt, is_border_particle):
    diagnostics.time_poisson()
    no_particles = np.size(y)//(2+2+1)

    r_dot = np.zeros((no_particles, 2))
    v_dot = np.zeros((no_particles, 2))
    p_dot = np.zeros((no_particles, 1))

    r = y[: 2*no_particles]
    v = y[2*no_particles : 4*no_particles]
    p = y[4*no_particles :]
    r = r.reshape(-1, 2)
    v = v.reshape(-1, 2)
    p = p.reshape(-1, 1)

    for i, (r_i, v_i, p_i) in enumerate(zip(r, v, p)):
        if not is_border_particle[i]:
            r_dot[i] = np.zeros(2)
            v_dot[i], p_dot[i] = _pressure_gradient(r_i, v_i, p_i, r, v, p, dt)
        else:
            r_dot[i] = np.zeros(2)
            v_dot[i] = np.zeros(2)
            p_dot[i] = np.zeros(1)

    diagnostics.log_np_array(v_dot)
    diagnostics.log_np_array(p_dot)

    r_dot = r_dot.reshape(-1, order="C")
    v_dot = v_dot.reshape(-1, order="C")
    p_dot = p_dot.reshape(-1, order="C")
    y_dot = np.concatenate((r_dot, v_dot, p_dot))

    diagnostics.time_poisson()
    return y_dot

def _pressure_gradient(
    r_i: np.array,
    v_i: np.array,
    p_i: np.array,
    r: np.array,
    v: np.array,
    p: np.array,
    dt: float,
):
    result = np.zeros(2)
    coefficients = _solve_least_squares_ppe(r_i, v_i, p_i, r, v, p, dt)

    # TODO: find out if this really is how you are supposed to
    # calculate the gradient
    for j, r_j in enumerate(r):
        result[0] = (
            - coefficients[1]
            + coefficients[3]*(r_j[0] - r_i[0])
            - coefficients[4]*(r_i[1] - r_j[1])
        )
        result[1] = (
            - coefficients[2]
            + coefficients[4]*(r_j[1] - r_i[1])
            - coefficients[5]*(r_i[0] - r_j[0])
        )

        # TODO: check if this is the correct pressure update
        pressure = coefficients[3] + coefficients[5]

    diagnostics.log_np_array(result)
    return result, pressure

# TODO: add boundary condition
def _solve_least_squares_ppe(
    r_i: np.array,
    v_i: np.array,
    p_i: np.array,
    r: np.array,
    v: np.array,
    p: np.array,
    dt: float,
):
    D = []
    W = []
    b = []
    for j, (r_j, p_j) in enumerate(zip(r, p)):
        kernel = gauss(r_i, r_j)
        if kernel > 0:
            D.append([
                1,
                r_j[0] - r_i[0],
                r_j[1] - r_i[1],
                (r_j[0] - r_i[0])*(r_j[0] - r_i[0])*.5,
                (r_j[0] - r_i[0])*(r_j[1] - r_i[1]),
                (r_j[1] - r_i[1])*(r_j[1] - r_i[1])*.5,
            ])
            b.extend([p_j[0]])
            W.extend([np.sqrt(kernel)])

    # PPE
    D.append([1, 0, 0, 0, 0, 0])
    W.extend([1])
    nabla_operator = nabla(r_i, v_i, r, v)
    nabla_dot_product = (nabla_operator[0][0] + nabla_operator[1][1])/dt
    b.extend([nabla_dot_product])

    D = np.array(D)
    b = np.array(b)
    W = np.diag(W)

    coefficients_ppe = np.linalg.lstsq(-W@D, b)[0]

    diagnostics.log_np_array(coefficients_ppe)

    return coefficients_ppe



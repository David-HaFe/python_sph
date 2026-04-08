

import numpy as np

from utils.diagnostics import diagnostics
from model_parameters import model_parameters

# SPH smoothing length
h_default = 3

# gauss kernel for given point and reference point
def gauss(r_a: np.array, r_b: np.array, h=h_default):
    diagnostics.time_kernel()
    alpha = 1
    distance = np.linalg.norm(r_a-r_b)

    if distance < h:
        result = np.exp(-alpha * distance**2/h**2)
    else:
        result = 0

    diagnostics.time_kernel()
    return result

def _solve_least_squares(r_i: np.array, T_i, r: np.array, T:np.array):
    diagnostics.time_least_squares()
    D = []
    W = []
    delta_f = []
    for j, (r_j, T_j) in enumerate(zip(r, T)):
        gaussian = gauss(r_i, r_j)
        if gaussian > 0:
            diagnostics.register_particle(True)
            D.append([
                r_j[0] - r_i[0],
                r_j[1] - r_i[1],
                (r_j[0] - r_i[0])*(r_j[0] - r_i[0])*.5,
                (r_j[0] - r_i[0])*(r_j[1] - r_i[1]),
                (r_j[1] - r_i[1])*(r_j[1] - r_i[1])*.5,
            ])
            delta_f.extend([T_i - T_j])
            W.extend([np.sqrt(gaussian)])
        else:
            diagnostics.register_particle(False)

    D = np.array(D)
    delta_f = np.array(delta_f)
    W = np.diag(W)

    coefficients = np.linalg.lstsq(-W@D, delta_f)[0]

    diagnostics.time_least_squares()
    return coefficients

def nabla(r_a:np.array, T_a, r:np.array, T:np.array):
    diagnostics.time_nabla()

    diagnostics.time_nabla()

def laplace(r_a:np.array, T_a, r:np.array, T:np.array):
    diagnostics.time_laplace()

    coefficients = _solve_least_squares(r_a, T_a, r, T)
    result = np.array([-coefficients[2], -coefficients[4]])
    result = (coefficients[2] + coefficients[4])
    diagnostics.time_laplace()
    return result



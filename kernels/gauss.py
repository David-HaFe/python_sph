

import numpy as np

from utils.diagnostics import diagnostics

# SPH smoothing length
h_default = 3

# gauss kernel for given point and reference point
def gauss(r_i: np.array, r_j: np.array, h=h_default):
    diagnostics.time_kernel()
    alpha = 1
    distance = np.linalg.norm(r_i-r_j)

    if distance < h:
        result = np.exp(-alpha * distance**2/h**2)
    else:
        result = 0

    diagnostics.time_kernel()
    return result

def _solve_least_squares(
    r_i: np.array,
    function_i: np.array,
    r: np.array,
    function:np.array
):
    diagnostics.time_least_squares()
    D = []
    W = []
    delta_f = []
    for j, (r_j, function_j) in enumerate(zip(r, function)):
        kernel = gauss(r_i, r_j)
        if kernel > 0:
            diagnostics.register_particle(True)
            D.append([
                r_j[0] - r_i[0],
                r_j[1] - r_i[1],
                (r_j[0] - r_i[0])*(r_j[0] - r_i[0])*.5,
                (r_j[0] - r_i[0])*(r_j[1] - r_i[1]),
                (r_j[1] - r_i[1])*(r_j[1] - r_i[1])*.5,
            ])
            delta_f.extend([function_i - function_j])
            W.extend([np.sqrt(kernel)])
        else:
            diagnostics.register_particle(False)

    D = np.array(D)
    delta_f = np.array(delta_f)
    W = np.diag(W)

    coefficients = np.linalg.lstsq(-W@D, delta_f)[0]

    diagnostics.time_least_squares()
    return coefficients

def nabla(
    r_i:np.array,
    function_i:np.array,
    r_j:np.array,
    function_j:np.array,
    r:np.array,
    function:np.array
):
    diagnostics.time_nabla()

    coefficients = _solve_least_squares(r_i, function_i, r, function)
    result = np.array([
        coefficients[0]
        - coefficients[2]*(r_j[0] - r_i[0])
        + coefficients[3]*(r_i[1] - r_j[1]),
        coefficients[1]
        - coefficients[4]*(r_j[1] - r_i[1])
        + coefficients[3]*(r_i[0] - r_j[0]),
    ])

    diagnostics.time_nabla()
    return result

def laplace(
    r_i:np.array,
    function_i:np.array,
    r:np.array,
    function:np.array,
):
    diagnostics.time_laplace()

    coefficients = _solve_least_squares(r_i, function_i, r, function)
    result = (coefficients[2] + coefficients[4])
    diagnostics.time_laplace()
    return result



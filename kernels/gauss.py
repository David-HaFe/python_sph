import numpy as np
import scipy as sp

from utils.diagnostics import diagnostics
from config import kernel_length

alpha = 6.25  # don't change this, otherwise kernel looks not very good
# compute this once for speed
default_kernel_coefficient = -alpha/(kernel_length**2)

# gauss kernel for given point and reference point
def gauss(r_i: np.array, r_j: np.array, h=kernel_length):
    # diagnostics.time_kernel()
    distance = np.linalg.norm(r_i - r_j)

    if h == kernel_length:
        kernel_coefficient = default_kernel_coefficient
    else:
        kernel_coefficient = -alpha / (h**2)

    if distance < h:
        result = np.exp(kernel_coefficient * distance**2)
        # diagnostics.register_particle(True)
    else:
        result = 0
        # diagnostics.register_particle(False)

    # diagnostics.time_kernel()
    return result


def _solve_least_squares_gauss(
    r_i: np.array,
    function_i: np.array,
    r: np.array,
    function: np.array,
):
    # diagnostics.time_least_squares()

    D = []
    W = []
    b = []
    for j, (r_j, function_j) in enumerate(zip(r, function)):
        kernel = gauss(r_i, r_j)
        if kernel > 0:
            D.append(
                [
                    r_j[0] - r_i[0],
                    r_j[1] - r_i[1],
                    (r_j[0] - r_i[0]) * (r_j[0] - r_i[0]) * 0.5,
                    (r_j[0] - r_i[0]) * (r_j[1] - r_i[1]),
                    (r_j[1] - r_i[1]) * (r_j[1] - r_i[1]) * 0.5,
                ]
            )
            b.extend([function_i - function_j])
            W.extend([np.sqrt(kernel)])

    D = np.array(D)
    b = np.array(b)
    W = np.array(W)
    # W = np.diag(W)

    coefficients = np.linalg.lstsq(-W[:, None] * D, b)[0]
    # coefficients = np.linalg.lstsq(-W @ D, b)[0]

    # diagnostics.time_least_squares()
    return coefficients


def nabla(
    r_i: np.array,
    function_i: np.array,
    r: np.array,
    function: np.array,
):
    # diagnostics.time_nabla()

    result = np.zeros((2, len(function_i)))
    coefficients = _solve_least_squares_gauss(r_i, function_i, r, function)

    # TODO: find out if this really is how you are supposed to
    # calculate the gradient

    for j, r_j in enumerate(r):
        result[0] = (
            coefficients[0]
            + coefficients[2] * (r_j[0] - r_i[0])
            + coefficients[3] * (r_j[1] - r_i[1])
        )
        result[1] = (
            coefficients[1]
            + coefficients[4] * (r_j[1] - r_i[1])
            + coefficients[3] * (r_j[0] - r_i[0])
        )

    # diagnostics.time_nabla()
    return result


def laplace(
    r_i: np.array,
    function_i: np.array,
    r: np.array,
    function: np.array,
):
    # diagnostics.time_laplace()

    coefficients = _solve_least_squares_gauss(r_i, function_i, r, function)
    result = coefficients[2] + coefficients[4]

    # diagnostics.time_laplace()
    return result

import numpy as np
import scipy as sp

from utils.diagnostics import diagnostics
from config import (
    kernel_length,
    no_particles,
    use_neumann,
    set_neumann,
    border,
    dt,
)

alpha = 6.25  # don't change this, otherwise kernel looks not very good
# compute this once for speed
default_kernel_coefficient = -alpha / (kernel_length**2)


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
    r_i: np.array = None,
    v_i: np.array = None,
    function_i: np.array = None,
    r: np.array = None,
    v: np.array = None,
    function: np.array = None,
    add_incompressibility: bool = False,
):
    # diagnostics.time_least_squares()

    D = np.zeros((no_particles, 5))
    W = np.zeros(no_particles)
    b = np.zeros(no_particles)
    count = 0

    for j, (r_j, function_j) in enumerate(zip(r, function)):
        kernel = gauss(r_i, r_j)
        if kernel > 0 and not((r_i[0] == r_j[0]) and (r_i[1] == r_j[1])):
            dr = r_j - r_i
            D[count] = [dr[0], dr[1], 0.5 * dr[0] ** 2, dr[0] * dr[1], 0.5 * dr[1] ** 2]
            W[count] = kernel
            b[count] = np.squeeze(function_i - function_j)
            count += 1

    # append neumannn boundary condition, since we found a border point
    if use_neumann and ((abs(r_i[0]) == border) or (abs(r_i[1]) == border)):

        # this may be two dimensional in case of an edge point
        normal_vector = set_neumann(r_i[0], r_i[1])
        no_boundary_conditions = normal_vector.shape[0]
        for i in range(no_boundary_conditions):
            D[count] = [normal_vector[i][0], normal_vector[i][1], 0, 0, 0]
            W[count] = 1
            b[count] = 0
            count += 1

    D = D[:count]

    # TODO: make sure that neumann border is handled properly as well,
    #       since the leftmost entries should be 0, and not 1
    # PPE
    if add_incompressibility:
        D = np.hstack([np.ones((D.shape[0], 1)), D])
        D = np.vstack([D, [0, 0, 0, 1, 0, 1]])
        W[count] = 1
        nabla_dot_product = (
            nabla(
                r_i=r_i,
                function_i=v_i[0],
                r=r,
                function=v[:, 0],
                add_incompressibility=False,
            )[0]
            + nabla(
                r_i=r_i,
                function_i=v_i[1],
                r=r,
                function=v[:, 1],
                add_incompressibility=False,
            )[1]
        )

        b[count] = nabla_dot_product
        count += 1

    W = W[:count]
    b = b[:count]

    D_transpose_W = D.T * W[None, :]
    try:
        coefficients = np.linalg.solve(-D_transpose_W @ D, D_transpose_W @ b)
    except np.linalg.LinAlgError as e:
        diagnostics.log_string("Crikey the program crashed")
        diagnostics.log_np_array(D)
        diagnostics.log_np_array(W)
        diagnostics.log_np_array(b)
        raise e

    # coefficients = np.linalg.lstsq(-W[:, None] * D, b)[0]
    # coefficients = np.linalg.lstsq(-W @ D, b)[0]

    # check if gradient is too steep
    # if use_neumann;
    if True:
        if abs(r_i[0]) == border:
            diagnostics.log_full_np_array(r_i)
            diagnostics.log_full_np_array(D)
            diagnostics.log_full_np_array(W)
            diagnostics.log_full_np_array(b)
            diagnostics.log_full_np_array(coefficients)
            assert abs(coefficients[0]) < 1e-7
        if abs(r_i[1]) == border:
            diagnostics.log_full_np_array(r_i)
            diagnostics.log_full_np_array(D)
            diagnostics.log_full_np_array(W)
            diagnostics.log_full_np_array(b)
            diagnostics.log_full_np_array(coefficients)
            assert abs(coefficients[1]) < 1e-7

    # diagnostics.time_least_squares()
    return coefficients


def nabla(
    r_i: np.array = None,
    v_i: np.array = None,
    function_i: np.array = None,
    r: np.array = None,
    v: np.array = None,
    function: np.array = None,
    add_incompressibility: bool = False,
):
    # diagnostics.time_nabla()

    result = np.zeros(2)

    coefficients = _solve_least_squares_gauss(
        r_i=r_i,
        v_i=v_i,
        function_i=function_i,
        r=r,
        v=v,
        function=function,
        add_incompressibility=add_incompressibility,
    )

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


# # returns the nabla operator, but as a scalar product with the function
# # basically just a wrapper
# def nabla_product(
#     r_i: np.array = None,
#     v_i: np.array = None,
#     function_i: np.array = None,
#     r: np.array = None,
#     v: np.array = None,
#     function: np.array = None,
#     add_incompressibility: bool = False,
# ):
#     nabla_result = nabla(
#         r_i=r_i,
#         v_i=v_i,
#         function_i=function_i,
#         r=r,
#         v=v,
#         function=function,
#         add_incompressibility=add_incompressibility,
#     )
#     return nabla_result[0] + nabla_result[1]


def laplace(
    r_i: np.array = None,
    v_i: np.array = None,
    function_i: np.array = None,
    r: np.array = None,
    v: np.array = None,
    function: np.array = None,
    add_incompressibility: bool = False,
):
    # diagnostics.time_laplace()

    coefficients = _solve_least_squares_gauss(
        r_i=r_i,
        v_i=v_i,
        function_i=function_i,
        r=r,
        v=v,
        function=function,
        add_incompressibility=add_incompressibility,
    )

    result = coefficients[2] + coefficients[4]

    # diagnostics.time_laplace()
    return result

def combo_deal(
    r_i: np.array = None,
    v_i: np.array = None,
    function_i: np.array = None,
    r: np.array = None,
    v: np.array = None,
    function: np.array = None,
    add_incompressibility: bool = False,
):
    nabla_operator = np.zeros(2)

    coefficients = _solve_least_squares_gauss(
        r_i=r_i,
        v_i=v_i,
        function_i=function_i,
        r=r,
        v=v,
        function=function,
        add_incompressibility=add_incompressibility,
    )

    for j, r_j in enumerate(r):
        nabla_operator[0] = (
            coefficients[0]
            + coefficients[2] * (r_j[0] - r_i[0])
            + coefficients[3] * (r_j[1] - r_i[1])
        )
        nabla_operator[1] = (
            coefficients[1]
            + coefficients[4] * (r_j[1] - r_i[1])
            + coefficients[3] * (r_j[0] - r_i[0])
        )

    laplace_operator = coefficients[2] + coefficients[4]

    return nabla_operator, laplace_operator

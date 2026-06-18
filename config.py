# true several constants etc. that should be used throughout the model

import numpy as np
from dataclasses import dataclass
from math import pi, sin
from utils.diagnostics import diagnostics

manufactured_solution_no = "manufactured_solutions.solution_3"

### grid ######################################################################
# number of particles in the x and y dimension
no_particles_x = 11
no_particles_y = no_particles_x

# interval where the x and y dimension are contained
# the result looks like this
# [-border,+border]             [+border,+border]
#                  \           /
#                   •    A    ·
#                        |
#                   -----+---->
#                        |
#                   ·    |    ·
#                  /           \
# [-border,-border]             [+border,-border]
border = 2

# choose which boundary condition type to choose
use_dirichlet = False
use_neumann = True

# assert use_neumann != use_dirichlet

# number of layers that the border has, spaced with same spacing as
# inside particles
if use_dirichlet:
    border_thickness = 1
else:
    # DO NOT TOUCH
    # border has to be 0 if dirichlet is not used
    border_thickness = 0


# neumann boundary definition
def set_neumann(x, y):

    assert (abs(x) == border) or (abs(y) == border)

    # at x border
    if abs(x) > abs(y):
        result = np.array([[np.sign(x), 0]])

    # at y border
    elif abs(x) < abs(y):
        result = np.array([[0, np.sign(y)]])

    # at edge
    else:
        result = np.array(
            [
                [np.sign(x), 0],
                [0, np.sign(y)],
            ]
        )

    return result


# can be used to throw an arbitrary border condition onto the border
def set_dirichlet(x, y):
    return 0
    # bottom or left border, set to 0
    # if (x <= -border) or (y <= -border):
    #     result = 0.0
    #
    # # edge at then end, set to 1
    # elif (x > border) and (y > border):
    #     result = 1.0
    #
    # # border on the right, put sine there
    # elif x > border:
    #     result = sin(pi / (4 * border) * y + pi / 4)
    #
    # # border at the top, put another sine there
    # elif y > border:
    #     result = sin(pi / (4 * border) * x + pi / 4)
    #
    # # this should never occur
    # else:
    #     result = -100
    #
    # diagnostics.log_string(f"at {x},{y}: {result}")
    # return result


# DO NOT TOUCH
# positions without border, spacing between particles
x_positions, dx = np.linspace(-border, border, no_particles_x, retstep=True)
y_positions, dy = np.linspace(-border, border, no_particles_y, retstep=True)

# DO NOT TOUCH
# number of total particles
no_particles = (no_particles_x + 2 * border_thickness) * (
    no_particles_y + 2 * border_thickness
)

### time scale and solver #####################################################
# start time
t0 = 0.0

# end time
t1 = 10.0

# number of steps
steps_per_sec = 20

# playback speed of the animation
playback_speed = 1

# DO NOT TOUCH
# number of time steps and step size
no_steps = int(steps_per_sec * (t1 - t0)) + 1
dt = (t1 - t0) / no_steps

# # calculate how far apart particles are
# spacing_x = border * 2 / (no_particles_x - 1)
# spacing_y = border * 2 / (no_particles_y - 1)

### kernel ####################################################################
# factor by which kernel support should exceed initial grid distance
kernel_scaling = 1.5

# DO NOT TOUCH
# actual kernel support length based on kernel scaling
kernel_length = kernel_scaling * (dx + dy) / 2

### physical properties #######################################################
# heat dissipation constant
heat_alpha = 0.2

# mass
m = 1
masses = np.full(no_particles, m)

# density
rho = 1  # -> chosen to be 1 by Tiwari Kuhnert (fpm)
densities = np.full(no_particles, rho)

# kinetic viscosity
nu = 1 / rho
kinetic_viscosities = np.full(no_particles, nu)

gravity = np.array([0, -0.81])

### visualization #############################################################
# for compare command
# specify all files that should be compared with scatter and mse here
# .npz shall not be written out
compared_files = np.array(
    [
        # stuff
        # "heat_equation/solutions/solution_16x16_r1_5_dt0_05",
        # "heat_equation_manufactured/solutions/solution_16x16_r1_5_dt0_05",

        # step
        # "heat_equation/solutions/solution_11x11_r1_5_dt0_1",
        # "heat_equation_manufactured/solutions/solution_11x11_r1_5_dt0_1",
        # "heat_equation/solutions/solution_11x11_r1_5_dt0_05",
        # "heat_equation_manufactured/solutions/solution_11x11_r1_5_dt0_05",
        # "heat_equation/solutions/solution_11x11_r1_5_dt0_025",
        # "heat_equation_manufactured/solutions/solution_11x11_r1_5_dt0_025",
        # "heat_equation/solutions/solution_11x11_r1_5_dt0_0125",
        # "heat_equation_manufactured/solutions/solution_11x11_r1_5_dt0_0125",

        # grid
        # "heat_equation/solutions/solution_6x6_r1_5_dt0_05",
        # "heat_equation_manufactured/solutions/solution_6x6_r1_5_dt0_05",
        # "heat_equation/solutions/solution_11x11_r1_5_dt0_05",
        # "heat_equation_manufactured/solutions/solution_11x11_r1_5_dt0_05",
        # "heat_equation/solutions/solution_21x21_r2_0_dt0_05",
        # "heat_equation_manufactured/solutions/solution_21x21_r2_0_dt0_05",

        # kernel
        "heat_equation/solutions/solution_11x11_r1_5_dt0_05",
        "heat_equation_manufactured/solutions/solution_11x11_r1_5_dt0_05",
        "heat_equation/solutions/solution_11x11_r2_0_dt0_05",
        "heat_equation_manufactured/solutions/solution_11x11_r2_0_dt0_05",
        "heat_equation/solutions/solution_11x11_r3_0_dt0_05",
        "heat_equation_manufactured/solutions/solution_11x11_r3_0_dt0_05",
        "heat_equation/solutions/solution_11x11_r4_5_dt0_05",
        "heat_equation_manufactured/solutions/solution_11x11_r4_5_dt0_05",
    ]
)

# points at which the solution should be compared (in seconds)
snapshots = np.array(
    [
        0.1,
        1,
        3,
        8,
        14,
    ]
)


# for visualize kernel command
# options: "gauss", "wendland", everything you decide to add (:
kernel_choice = "gauss"


# DO NOT TOUCH - recomputes values after no_particles has been changed
def recompute():
    # positions without border, spacing between particles
    x_positions, dx = np.linspace(-border, border, no_particles_x, retstep=True)
    y_positions, dy = np.linspace(-border, border, no_particles_y, retstep=True)

    # number of total particles
    no_particles = (no_particles_x + 2 * border_thickness) * (
        no_particles_y + 2 * border_thickness
    )

    # actual kernel support length based on kernel scaling
    kernel_length = kernel_scaling * (dx + dy) / 2


@dataclass
class sim_run_parameters:
    is_analytical: bool
    is_manufactured: bool
    no_particles: int
    kernel_radius: float
    step_size: float


### sim result data class #####################################################
@dataclass
class sim_result:
    t: np.array
    x: np.ndarray
    y: np.ndarray
    data_1: np.ndarray
    data_2: np.ndarray
    is_border_particle: np.array

# several constants etc. that should be used throughout the model

import numpy as np
from dataclasses import dataclass

### grid ######################################################################
# number of particles in the x dimension
no_particles_x = 10

# number of particles in the y direction
no_particles_y = 10

# interval where the x and y dimension are contained
# the result looks like this
#                  +border
#         x    A    x +border
#              |
#         -----+---->
#              |
# -border x    |    x
#      -border
border = 2

# DO NOT TOUCH
# positions without border, spacing between particles
x_positions, dx = np.linspace(-border, border, no_particles_x, retstep=True)
y_positions, dy = np.linspace(-border, border, no_particles_y, retstep=True)

# number of layers that the border has, spaced with same spacing as
# inside particles
border_thickness = 0

# DO NOT TOUCH
# number of total particles
no_particles = (no_particles_x + 2 * border_thickness) * (
    no_particles_y + 2 * border_thickness
)

### time scale and solver #####################################################
# start time
t0 = 1.0

# end time
t1 = 31.0

# number of steps
no_steps = 300

# DO NOT TOUCH
# number of time steps
dt = (t1 - t0) / no_steps

# # calculate how far apart particles are
# spacing_x = border * 2 / (no_particles_x - 1)
# spacing_y = border * 2 / (no_particles_y - 1)

### kernel ####################################################################
# factor by which kernel support should exceed initial grid distance
kernel_scaling = 1.5

# DO NOT TOUCH
# actual kernel support length based on kernel scaling
kernel_length = kernel_scaling * (spacing_x + spacing_y) / 2

### physical properties #######################################################
# heat dissipation constant
heat_alpha = 0.05

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
        # "heat_equation_analytical/solutions/solution_3x3_r1_5",
        "heat_equation_analytical/solutions/solution_5x5_r1_5",
        "heat_equation_analytical/solutions/solution_10x10_r1_5",
        "heat_equation_analytical/solutions/solution_15x15_r1_5",
        # "heat_equation_analytical/solutions/solution_20x20_r1_5",
        # "heat_equation/solutions/solution_3x3_r1_5",
        "heat_equation/solutions/solution_5x5_r1_5",
        "heat_equation/solutions/solution_10x10_r1_5",
        "heat_equation/solutions/solution_15x15_r1_5",
        # "heat_equation/solutions/solution_20x20_r1_5
    ]
)

# for visualize kernel command
# options: "gauss", "wendland", everything you decide to add (:
kernel_choice = "gauss"

### sim result data class #####################################################
@dataclass
class sim_result:
    t: np.array
    x: np.ndarray
    y: np.ndarray
    data_1: np.ndarray
    data_2: np.ndarray
    is_border_particle: np.array

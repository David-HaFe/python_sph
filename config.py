# several constants etc. that should be used throughout the model

import numpy as np

no_particles_x = 5
no_particles_y = 5
border = 2
# positions without border
x_positions, dx = np.linspace(-border, border, no_particles_x, retstep=True)
y_positions, dy = np.linspace(-border, border, no_particles_y, retstep=True)

border_thickness = 0
no_particles = (no_particles_x + 2 * border_thickness) * (
    no_particles_y + 2 * border_thickness
)

# time scale and solver
t0 = 1.0
t1 = 31.0
no_steps = 300
dt = (t1 - t0) / no_steps

# calculate how far apart particles are
spacing_x = border * 2 / (no_particles_x - 1)
spacing_y = border * 2 / (no_particles_y - 1)

# kernel
kernel_scaling = 1.5
kernel_length = kernel_scaling * (spacing_x + spacing_y) / 2

# heat equation
heat_alpha = 0.05

# physical properties
m = 1
masses = np.full(no_particles, m)
rho = 1  # -> chosen to be 1 by Tiwari Kuhnert (fpm)
densities = np.full(no_particles, rho)
nu = 1 / rho
kinetic_viscosities = np.full(no_particles, nu)
gravity = np.array([0, -0.81])

# for compare command
# .csv shall not be written out
compared_files = np.array(
    [
        "heat_equation/solutions/solution_5x5_r1_5",
        "heat_equation_analytical/solutions/solution_5x5_r1_5",
        # "heat_equation/solutions/solution_10x10_r1_5",
        # "heat_equation/solutions/solution_15x15_r1_5",
        # "heat_equation_analytical/solutions/solution_10x10_r1_5",
        # "heat_equation_analytical/solutions/solution_15x15_r1_5",
        # "heat_equation/solutions/solution_10x10_r1_5",
        # "heat_equation/solutions/solution_15x15_r1_5",
        # "heat_equation/solutions/solution_20x20_r1_5
    ]
)

# for visualize kernel command
# options: "gauss", "wendland", everything you decide to add (:
kernel_choice = "gauss"

# class Model_Parameters:
#     def __init__(self):
#         self.gravity = np.array([0, -0.81])
#         self._unified_mass = 1
#         self._unified_density = 1  # chosen by tiwari kuhnert (fpm)
#         self._unified_kinetic_viscosity = 1 / self._unified_density
#         self.m = np.array([])
#         self.nu = np.array([])
#         self.rho = np.array([])
#
#     def set_no_particles(self, no_particles):
#         self.m = np.full(no_particles, self._unified_mass)
#         self.nu = np.full(no_particles, self._unified_kinetic_viscosity)
#         self.rho = np.full(no_particles, self._unified_density)
#
#
# model_parameters = Model_Parameters()

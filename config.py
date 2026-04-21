# several constants etc. that should be used throughout the model

import numpy as np

no_particles_x = 15
no_particles_y = 15
border = 5
# positions without border
x_positions, dx = np.linspace(-border, border, no_particles_x, retstep=True)
y_positions, dy = np.linspace(-border, border, no_particles_y, retstep=True)

border_thickness = 2
no_particles = (no_particles_x + 2 * border_thickness) * (
    no_particles_y + 2 * border_thickness
)

t0 = 0.0
t1 = 30.0
no_steps = 300
dt = (t1 - t0) / no_steps

# calculate how far apart particles are,
# then
spacing_x = border * 2 / (no_particles_x - 1)
spacing_y = border * 2 / (no_particles_y - 1)
kernel_scaling = 3
kernel_length = kernel_scaling * (spacing_x + spacing_y) / 2


class Model_Parameters:
    def __init__(self):
        self.gravity = np.array([0, -0.81])
        self.eta = 0.1
        self._unified_mass = 1
        self._unified_density = 1  # chosen by tiwari kuhnert (fpm)
        self._unified_kinetic_viscosity = 1 / self._unified_density
        self.m = np.array([])
        self.nu = np.array([])
        self.rho = np.array([])

    def set_no_particles(self, no_particles):
        self.m = np.full(no_particles, self._unified_mass)
        self.nu = np.full(no_particles, self._unified_kinetic_viscosity)
        self.rho = np.full(no_particles, self._unified_density)


model_parameters = Model_Parameters()

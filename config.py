

# several constants etc. that should be used throughout the model

import numpy as np

no_particles_x = 20
no_particles_y = 20
border = 5
# positions without border
x_positions, dx = np.linspace(-border, border, no_particles_x, retstep=True)
y_positions, dy = np.linspace(-border, border, no_particles_y, retstep=True)

border_thickness = 2
no_particles = (
    no_particles_x+2*border_thickness)*(no_particles_y+2*border_thickness
)
kernel_length = 1.5

class Model_Parameters():
    def __init__(self):
        self.gravity = np.array([0, -0.81])
        self.eta = .1
        self._unified_mass = 1
        self._unified_density = 1 # chosen by tiwari kuhnert (fpm)
        self._unified_kinetic_viscosity = 1/self._unified_density
        self.m = np.array([])
        self.nu = np.array([])
        self.rho = np.array([])

    def set_no_particles(self, no_particles):
        self.m = np.full(no_particles, self._unified_mass)
        self.nu = np.full(no_particles, self._unified_kinetic_viscosity)
        self.rho = np.full(no_particles, self._unified_density)

model_parameters = Model_Parameters()



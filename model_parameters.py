

# several constants etc. that should be used throughout the model

import numpy as np

class Model_Parameters():
    def __init__(self):
        self.gravity = np.array([0, -9.81])
        self.eta = .1
        self._unified_mass = 1
        self._unified_density = 1
        self._unified_kinetic_viscosity = 1/self._unified_density
        self.m = np.array([])
        self.nu = np.array([])
        self.rho = np.array([])

    def set_no_particles(self, no_particles):
        self.m = np.full(no_particles, self._unified_mass)
        self.nu = np.full(no_particles, self._unified_kinetic_viscosity)
        self.rho = np.full(no_particles, self._unified_density)

model_parameters = Model_Parameters()



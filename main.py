

# main file for the implementation of sph
# author: David Hambach Ferrer

# import numpy as np
import jax.numpy as jnp
import scipy as sp
import matplotlib as plt

from particle import Particle

# constants
rho_a = 1
kernel_radius = 2
h = 1   # -> SPH smoothing length
particles = np.empty([])
gravity = np.array([0,-9.81])

# initial condition




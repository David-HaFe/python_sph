

# main file for the implementation of sph
# author: David Hambach Ferrer

# import numpy as np
import numpy as np
import jax.numpy as jnp
from scipy.integrate import odeint
import matplotlib as plt
from equations import navier_stokes

# constants
rho_a = 1
gravity = np.array([0, -9.81])
no_particles = 4

# initialize grid
positions = np.empty([])
velocities = np.empty([])

for i in range(0, 3):
    np.append(positions, jnp.array([i, i]))
    np.append(velocities, jnp.array([-i, -i]))

masses = np.full(no_particles, 1.0)
densities = np.full(no_particles, 2.0)
pressures = np.full(no_particles, 1.0)
dynamic_viscosities = np.full(no_particles, 1.0)

# simulation
t = np.linspace(0,1,100)
solution = odeint(
    lambda q, t: navier_stokes(q,
                               t,
                               masses,
                               densities,
                               pressures,
                               dynamic_viscosities),
    np.array([positions, velocities]),
    t,
)

print(solution)



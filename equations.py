

# equations needed for the sph implementation

# import numpy as np
import jax.numpy as jnp
from particle import Particle

# implements the discretized navier stokes equation for the ode solver
def navier_stokes(t, q, particle_a):
    # extract values from particle a that are needed for all calculations
    pressure_term = 0
    viscosity_term = 0
    x_a = particla_a.x()
    v_a = particle_a.v()
    p_a= particle_a.p()
    rho_a = particle_a.rho()
    pressure_contribution_a = p_a/jnp.power(rho_a, 2)
    mu_a = particle_a.mu()

    for particle in jnp.nditer(particles):
        pressure_term += particle.m() * (
                particle.p()/jnp.power(particle_a.rho(), 2)
                + pressure_contribution_a
                ) * delta_W(x_a, particle.x())

        delta_x = x_a - particle.x()
        delta_v = v_a - particle.v()

        viscosity_term += particle.m() *
            (mu_a + particle.mu())/(rho_a*particle.rho())
            *(delta_x*delta_v)/(jnp.power(delta_x, 2) + .01*h**2)
            * delta_W(x_a, particle.x())


    return rho_a * (pressure_term+viscosity_term)

# kernel for a given point
# iterate over each particle and collect contributions
# using Wendland C² kernel
def W(x_a: jnp.array):
    kernel_result = 0

    for particle in jnp.nditer(particles):
        distance = jnp.linalg.norm(x_a - particle.x())/h

        if (distance < kernel_radius):
            kernel_result +=
            sigma_W/h_dim * (1 + 2*distance)*jnp.power((1 - .5*distance), 4)

        return kernel_result

# gradient of the kernel function
def delta_W(x_a: jnp.array, x_b: jnp.array):
    return jax.grad(W(x_a), x_b)



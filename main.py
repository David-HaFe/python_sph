

# main file for the implementation of sph
# author: David Hambach Ferrer

# import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import random as rnd

from integrator import euler_forward
from equations import navier_stokes
from diagnostics import diagnostics
from animation import animate_solution

# initialize grid
initial_condition = np.array([])
is_wall_particle = []

x_limit = 5
y_limit = 5
spacing = 1
wall_spacing = .1
no_particles = x_limit * y_limit

def noise(): return -.1 + .2*rnd.random()

x_0 = []
v_0 = []
rho_0 = []

for x in range(0, x_limit):
    for y in range(0, y_limit):
        x_0.extend([x*spacing + noise(), y*spacing])
        v_0.extend([0, 0])
        rho_0.extend([1])

        # add wall particle flag
        is_wall_particle.extend([False])

dummy_density = .1
# initialize wall particles
for y in range(-1, int(y_limit/wall_spacing)):
    # left wall
    x_0.extend([-1, y*spacing*wall_spacing-1])
    v_0.extend([0, 0])
    rho_0.extend([dummy_density])
    is_wall_particle.extend([True])

    # right wall
    x_0.extend([spacing*x_limit, y*spacing*wall_spacing-1])
    v_0.extend([0, 0])
    rho_0.extend([dummy_density])
    is_wall_particle.extend([True])
    no_particles += 2

for x in range(-1, int((x_limit+1)/wall_spacing)):
    # bottom wall
    x_0.extend([x*spacing*wall_spacing-1, -1])
    v_0.extend([0, 0])
    rho_0.extend([dummy_density])
    is_wall_particle.extend([True])
    no_particles += 1

x_0 = np.array(x_0, dtype=float)
v_0 = np.array(v_0, dtype=float)
rho_0 = np.array(rho_0, dtype=float)
y_0 = np.concatenate((x_0, v_0, rho_0))
is_wall_particle = np.array(is_wall_particle)

print(np.size(x_0))
print(np.size(v_0))
print(np.size(rho_0))

# simulation
diagnostics.time_ode()

# sol = euler_forward(
#     function=navier_stokes,
#     initial_condition=y_0,
#     t_start=0,
#     t_end=4,
#     dt=.01,
# )

t_0 = 0
t_1 = 2
t_span = (t_0, t_1)
dt = 100
t_eval = np.linspace(t_0, t_1, num=dt)

sol = solve_ivp(
    fun = lambda t, y: navier_stokes(t, y, is_wall_particle),
    t_span = t_span,
    y0 = y_0,
    method = "RK23",
    rtol = 1e-3,
    atol = 1e-3,
    t_eval = t_eval,
)
print("")
print(sol.message)

diagnostics.time_ode()

diagnostics.print_diagnostics()
# animate_solution(sol.t, sol.y, x_0)

x_0 = x_0.reshape(-1, 2)
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
scat = ax.scatter(x_0[:, 0], x_0[:, 1])
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def animate(i):
    y_i = sol.y[:, i]
    x_i = y_i[: 2*no_particles]
    x_i = x_i.reshape(-1, 2)
    time_text.set_text(f't = {sol.t[i]:.3f}')
    scat.set_offsets(x_i)
    return (scat,)

ani = animation.FuncAnimation(
    fig,
    animate,
    repeat=True,
    frames=len(sol.t),
    interval=5
)

plt.show()



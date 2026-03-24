

# main file for the implementation of sph
# author: David Hambach Ferrer

# import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from integrator import euler_forward
from equations import navier_stokes
from diagnostics import diagnostics
from animation import animate_solution

# initialize grid
initial_condition = np.array([])
is_wall_particle = []

x_limit = 2
y_limit = 2
no_particles = x_limit * y_limit

x_0 = []
v_0 = []
rho_0 = []

for x in range(0, x_limit):
    for y in range(0, y_limit):
        # position
        x_0.extend([x, y])

        # velocity
        v_0.extend([0, 0])

        # density
        rho_0.extend([1])

        # add wall particle flag
        is_wall_particle.extend([False])

x_0 = np.array(x_0, dtype=float)
v_0 = np.array(v_0, dtype=float)
rho_0 = np.array(rho_0, dtype=float)
y_0 = np.concatenate((x_0, v_0, rho_0))
is_wall_particle = np.array(is_wall_particle)

dummy_density = 100
# initialize wall particles
# for x in range(0, x_limit + 1):
#     # left wall
#     initial_condition = np.append(initial_condition, np.array([x, -1]))
#     # dummy values for velocity and density
#     initial_condition = np.append(
#         initial_condition, np.array([0, 0, dummy_density]))
#     is_wall_particle = np.append(is_wall_particle, True)
#
#     # right wall
#     initial_condition = np.append(initial_condition, np.array([x, y_limit+1]))
#     # dummy values for velocity and density
#     initial_condition = np.append(
#         initial_condition, np.array([0, 0, dummy_density]))
#     is_wall_particle = np.append(is_wall_particle, True)
#
# for y in range(-1, y_limit+1):
#     # bottom wall
#     initial_condition = np.append(initial_condition, np.array([-1, y]))
#     # dummy values for velocity and density
#     initial_condition = np.append(
#         initial_condition, np.array([0, 0, dummy_density]))
#     is_wall_particle = np.append(is_wall_particle, True)

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
    fun = navier_stokes,
    t_span = t_span,
    y0 = y_0,
    method = "RK23",
    rtol = 1e-3,
    atol = 1e-5,
    t_eval = t_eval,
)
print("")
diagnostics.time_ode()

diagnostics.print_diagnostics()
# animate_solution(sol.t, sol.y, x_0)

x_0 = x_0.reshape(-1, 2)
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
scat = ax.scatter(x_0[:, 0], x_0[:, 1])

def animate(i):
    y_i = sol.y[:, i]
    x_i = y_i[: 2*no_particles]
    x_i = x_i.reshape(-1, 2)
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



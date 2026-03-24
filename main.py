

# main file for the implementation of sph
# author: David Hambach Ferrer

# import numpy as np
import numpy as np
from scipy.integrate import solve_ivp

from integrator import solve
from equations import navier_stokes
from diagnostics import diagnostics
from animation import animate_solution
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initialize grid
x0 = []
u0 = []
rho0 = []
is_wall_particle = []
# initial_condition = np.array([])
# is_wall_particle = np.array([])

x_limit = 4
y_limit = 4
n_particle = x_limit * y_limit

for x in range(0, x_limit):
    for y in range(0, y_limit):
        # position
        # initial_condition = np.append(initial_condition, np.array([x, y]))
        x0.extend([x, y])

        # velocity
        # initial_condition = np.append(initial_condition, np.array([0, 0]))
        u0.extend([0, 0])

        # initial density
        # initial_condition = np.append(initial_condition, 1)
        rho0.append(1)

        # is_wall_particle = np.append(is_wall_particle, False)
        is_wall_particle.append(False)

x0 = np.array(x0, dtype=float)
u0 = np.array(u0, dtype=float)
rho0 = np.array(rho0, dtype=float)
y0 = np.concatenate((x0, u0, rho0))
is_wall_particle = np.array(is_wall_particle)

# visualize initial configuration
X0 = x0.reshape(-1, 2)
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
scat = ax.scatter(X0[:, 0], X0[:, 1])
# plt.show()

dummy_density = 100
# # initialize wall particles
# for x in range(-1, x_limit + 1):
#     # left wall
#     initial_condition = np.append(initial_condition, np.array([x, -1]))
#     # dummy values for velocity and density
#     initial_condition = np.append(
#         initial_condition, np.array([0, 0, dummy_density]))
#     is_wall_particle = np.append(is_wall_particle, True)

#     # right wall
#     initial_condition = np.append(initial_condition, np.array([x, y_limit+1]))
#     # dummy values for velocity and density
#     initial_condition = np.append(
#         initial_condition, np.array([0, 0, dummy_density]))
#     is_wall_particle = np.append(is_wall_particle, True)

# for y in range(-1, y_limit+1):
#     # bottom wall
#     initial_condition = np.append(initial_condition, np.array([-1, y]))
#     # dummy values for velocity and density
#     initial_condition = np.append(
#         initial_condition, np.array([0, 0, dummy_density]))
#     is_wall_particle = np.append(is_wall_particle, True)

# simulation
diagnostics.time_ode()
# sol = solve(
#     # function=lambda t, q: navier_stokes(t, q, is_wall_particle),
#     navier_stokes,
#     # initial_condition=initial_condition,
#     initial_condition=y0,
#     dt=.01,
# )

t0 = 0.0
t1 = 0.5
t_span = (t0, t1)
num = 100
t_eval = np.linspace(t0, t1, num=num)
sol = solve_ivp(
    fun=navier_stokes,
    t_span=t_span,
    y0=y0,
    method = "RK23",
    # method = "RK45",
    # method = "BDF",
    rtol = 1e-1,
    atol = 1e-1,
    t_eval=t_eval,
)
diagnostics.time_ode()

diagnostics.print_diagnostics()
# animate_solution(sol.t, sol.y)

def animate(i):
    yi = sol.y[:, i]
    xi = yi[:2 * n_particle]
    Xi = xi.reshape(-1, 2)
    # scat.set_offsets((Xi[:, 0] - X0[:, 0], Xi[:, 1] - X0[:, 1]))
    # scat.set_offsets((Xi[:, 0], Xi[:, 1]))
    scat.set_offsets(Xi)
    return (scat,)

ani = animation.FuncAnimation(fig, animate, repeat=True, frames=len(sol.t), interval=5)

plt.show()

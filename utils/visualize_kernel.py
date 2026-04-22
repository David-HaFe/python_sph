# allows to plot the kernel function

import numpy as np
import matplotlib.pyplot as plt

from kernels.gauss import gauss
from kernels.wendland import wendland
from utils.heat_map import heat_plot
from scipy.interpolate import griddata

length = 100
x0 = range(-length, length + 1)
y0 = range(-length, length + 1)
function_value = []

# set up
for x in x0:
    for y in y0:
        function_value.extend(
            [
                gauss(
                    r_i=np.zeros(2),
                    r_j=np.array([x, y]),
                    h=100,
                )
            ]
        )
        # function_value.extend([wendland(
        #     x_a=np.zeros(2),
        #     x_b=np.array([x, y]),
        #     h=70,
        # )])

x0 = np.array(x0, dtype=float)
y0 = np.array(y0, dtype=float)
function_value = np.array(function_value, dtype=float)

# plot the kernel as a surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
XI, YI = np.meshgrid(x0, y0)
z_min = np.nanmin(function_value)
z_max = np.nanmax(function_value)
ax.set_zlim(z_min, 1.2 * z_max)
ax.set_title("here is your kernel sire")
ZI = griddata(
    (XI.ravel(), YI.ravel()),
    function_value.ravel(),
    (XI, YI),
    method="cubic",
)
ax.plot_surface(XI, YI, ZI, cmap="viridis", edgecolor="none")
plt.show()

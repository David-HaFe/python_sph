

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata
from utils.diagnostics import diagnostics
from config import no_particles_x, no_particles_y, kernel_length

def heat_plot(t, x, y, T):
    diagnostics.time_surface_plot()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    grid_points = 50

    # xi, yi don't change, just calculate once and continue
    xi = np.linspace(x[:, 1].min(), x[:, 1].max(), grid_points)
    yi = np.linspace(y[:, 1].min(), y[:, 1].max(), grid_points)
    XI, YI = np.meshgrid(xi, yi)
    z_min = np.nanmin(T)
    z_max = np.nanmax(T)

    def update(frame):
        ax.cla()
        ax.set_zlim(z_min, 2*z_max)
        ax.set_title(f"t = {t[frame]:.2f}")
        ZI = griddata(
            (x[:, frame], y[:, frame]),
            T[:, frame],
            (XI, YI),
            method="cubic",
        )
        ax.plot_surface(XI, YI, ZI, cmap="viridis", edgecolor="none")
        sys.stdout.write(f"\r\033[Kplotting surface @ {t[frame]}")
        sys.stdout.flush()

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100)
    # ani.save("animation.gif", writer="pillow", fps=30)
    param = str(kernel_length).replace(".", "_")
    name = f"visualizations/heat_transfer_{no_particles_x}x{no_particles_y}_r{param}.mp4"
    ani.save(name, writer="ffmpeg", fps=30)
    diagnostics.time_surface_plot()



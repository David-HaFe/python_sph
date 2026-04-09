

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata

def heat_plot(t, x, y, T):
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
        ax.set_zlim(z_min, 10*z_max)
        ax.set_title(f"uiuiui sieht schon ganz gut aus t = {t[frame]:.2f}")
        ZI = griddata(
            (x[:, frame], y[:, frame]),
            T[:, frame],
            (XI, YI),
            method="cubic",
        )
        ax.plot_surface(XI, YI, ZI, cmap="viridis", edgecolor="none")

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100)
    # ani.save("animation.gif", writer="pillow", fps=30)
    ani.save("visualizations/heat_transfer.mp4", writer="ffmpeg", fps=30)
    ani.save("visualizations/heat_transfer.gif", writer="pillow", fps=30)



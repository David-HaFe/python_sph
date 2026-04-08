

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata

def heat_plot(t, x, y, T):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        xi = np.linspace(x[:, frame].min(), x[:, frame].max(), 100)
        yi = np.linspace(y[:, frame].min(), y[:, frame].max(), 100)
        XI, YI = np.meshgrid(xi, yi)
        ax.cla()
        ax.set_title(f"t = {t[frame]:.2f}")
        ZI = griddata((x[:, frame], y[:, frame]), T[:, frame], (XI, YI), method="cubic")
        ax.plot_surface(XI, YI, ZI, cmap="viridis", edgecolor="none")

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100)
    ani.save("animation.gif", writer="pillow", fps=30)
    plt.show()



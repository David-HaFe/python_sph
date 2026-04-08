

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata

def heat_plot(t, r, T):
    print(T.shape)
    r = r.reshape(-1, 2)
    T = T.reshape(-1, 1).squeeze()
    # [[x0, y0], [x1,y1], ...]

    x, y = r[:, 0], r[:, 1]
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    XI, YI = np.meshgrid(xi, yi)

    print(x.shape)
    print(y.shape)
    print(T.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()
        ZI = griddata((x, y), T[frame], (XI, YI), method="cubic")
        ax.plot_surface(XI, YI, ZI, cmap="viridis", edgecolor="none")

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100)
    ani.save("../animation.gif", writer="pillow", fps=30)



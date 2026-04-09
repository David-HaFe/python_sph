

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# this file allows to plot the positions of particles over time
def particle_positions(t, x, y, is_border_particle):

    fig, ax = plt.subplots()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    colors = np.where(is_border_particle, "red", "blue")
    scat = ax.scatter(x[:, 0], y[:, 0], c=colors)

    def update(frame):
        scat.set_offsets(np.c_[x[:, frame], y[:, frame]])
        ax.set_title(f"t = {t[frame]:.2f}")

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100)
    ani.save("visualizations/particle_positions.mp4", writer="ffmpeg", fps=30)
    ani.save("visualizations/particle_positions.gif", writer="pillow", fps=30)


